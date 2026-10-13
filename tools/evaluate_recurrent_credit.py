"""Discriminate the existing native recurrent-state credit and recovery path."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))


def run(args, result, record):
    import torch
    from safetensors.torch import save_file, load_file
    from transformers import DynamicCache
    from nanoai.torch_source import load_full_text, LowRankLinear, export_hybrid_state, restore_hybrid_state
    from nanoai.core import digest_bytes, file_digest
    from nanoai.memory import Memory, StaleState, canonical
    from nanoai.experience import Experience
    from nanoai.agent import DEFAULT_IDENTITY

    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(61414)
    preflight = json.loads(args.preflight.read_text())
    if (preflight.get('model') != 'Qwen3.5-2B-BF16' or preflight.get('effort') != 'none'
            or not preflight.get('parent_source')):
        raise ValueError('A sourced native launch preflight is required')
    source_files = json.loads(args.source_before.read_text())
    current_files = {p.name:{'bytes':p.stat().st_size, 'digest':file_digest(p)}
                     for p in sorted(args.source.iterdir()) if p.is_file()}
    if current_files != source_files:
        raise ValueError('The original model source manifest changed')
    episode=json.loads(args.experience.read_text())
    current=Experience(Memory(args.collection_state)).export(episode['id'])
    if current != episode or current['eligibility']['status'] != 'checked':
        raise ValueError('The actual training experience changed')
    target_id=episode['assessment']['replayable_targets']['target_id']
    answer=next(c['content'] for c in episode['candidates'] if c['id']==target_id)
    content='Use the observed notice to answer the question in one short sentence.\nObserved notice:\n'+'\n'.join(episode['observations'])+'\n\nQuestion:\n'+episode['question']
    model,tok=load_full_text(args.source)
    projection=model.model.layers[22].linear_attn.in_proj_qkv
    adapter=LowRankLinear(projection)
    model.model.layers[22].linear_attn.in_proj_qkv=adapter
    record('loaded',layers=len(model.model.layers),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    prompt=tok.apply_chat_template([{'role':'user','content':content}],enable_thinking=False,add_generation_prompt=True,tokenize=False)
    ids=tok.encode(prompt,add_special_tokens=False)
    prefix=tok.encode(prompt.split('\n\nQuestion:\n')[0]+'\n\n',add_special_tokens=False)
    while ids[:len(prefix)] != prefix: prefix.pop()
    cut=len(prefix)
    target=tok.encode(answer,add_special_tokens=False)+[tok.eos_token_id]
    if not 2 <= cut < len(ids) or len(ids)+len(target)>512:raise ValueError('Causal token boundary differs')
    x1=torch.tensor([ids[:cut]]);x2=torch.tensor([ids[cut:]+target[:-1]])
    y=torch.tensor(target);start=len(ids)-1-cut
    record('inputs',experience=episode['id'],chosen_outcome=episode['assessment']['observed_outcome'],prefix_tokens=cut,later_input_tokens=x2.numel(),answer_tokens=len(target),training_role='oracle-supervised correction from a checked failed choice')

    def prefix_cache(cache_type):
        cache=cache_type(config=model.config)
        model.model(x1,past_key_values=cache,use_cache=True)
        return cache

    def later(cache):
        hidden=model.model(x2,past_key_values=cache,use_cache=True).last_hidden_state
        logits=model.lm_head(hidden[:,start:,:]).float()[0]
        return torch.nn.functional.cross_entropy(logits,y),logits

    def isolate(cache, keep_recurrent):
        retained=None
        for i,layer in enumerate(cache.layers):
            if hasattr(layer,'keys') and layer.keys is not None:
                layer.keys=layer.keys.detach();layer.values=layer.values.detach()
            if hasattr(layer,'conv_states'):
                for key,value in layer.conv_states.items():layer.conv_states[key]=value.detach()
                for key,value in layer.recurrent_states.items():
                    if keep_recurrent and i==22:
                        retained=[]
                        # DynamicCache updates this tensor in place after reading
                        # it. Observe the original write through its autograd hook;
                        # the mutated tensor's .grad describes a different version.
                        value.register_hook(lambda gradient, sink=retained:
                            sink.append(gradient.detach().float().clone()))
                    else:layer.recurrent_states[key]=value.detach()
        return retained

    def live_basis(candidate):
        return digest_bytes(canonical({'candidate':candidate,'source_files':source_files,'episode':episode['id'],'prefix_ids':ids[:cut],'suffix_ids':ids[cut:],'operator':file_digest(Path(__file__).resolve().parents[1]/'src/nanoai/torch_source.py')}))

    if args.reload_only:
        saved=torch.load(args.out/'hybrid-state.pt',map_location='cpu',weights_only=True)
        weights=load_file(str(args.out/'adapter.safetensors'))
        with torch.no_grad():
            adapter.a.copy_(weights['a']);adapter.b.copy_(weights['b'])
        basis=live_basis(file_digest(args.out/'adapter.safetensors'))
        memory=Memory(args.out/'state')
        with memory.locked():memory._validate_snapshot(saved['source_snapshot'])
        cache=restore_hybrid_state(saved['cache'],model.config,basis=basis)
        with torch.no_grad():value,logits=later(cache)
        if value.item()!=saved['loss'] or not torch.equal(logits,saved['logits']):
            raise ValueError('Fresh full hybrid state changed the later calculation')
        failures={}
        for name,wrong in [('candidate','changed-candidate'),('source','changed-source')]:
            try:restore_hybrid_state(saved['cache'],model.config,basis=wrong)
            except ValueError:failures[name]=True
            else:raise ValueError('Changed basis was accepted')
        receipt=saved['receipt'];path=memory.root/receipt['path'];before=(path.read_bytes(),path.stat().st_ino,path.stat().st_mtime_ns)
        memory.recover();after=(path.read_bytes(),path.stat().st_ino,path.stat().st_mtime_ns)
        if before!=after:raise ValueError('Recovery repeated a completed publication')
        memory.put_source('notice','This notice has been replaced by a new observation.',expected_revision=1)
        try:
            with memory.locked():memory._validate_snapshot(saved['source_snapshot'])
        except StaleState:failures['actual_source_correction']=True
        else:raise ValueError('Corrected source accepted historical hybrid state')
        result.update(status='completed',exact_logits=True,exact_loss=True,once_only_artifact=True,rejections=failures)
        record('fresh-state-restored',loss=value.item(),cache_bytes=(args.out/'hybrid-state.pt').stat().st_size,basis=basis)
        return

    adapter.detach_parameters=False
    cache=prefix_cache(DynamicCache);kept=isolate(cache,True)
    adapter.detach_parameters=True
    value,logits=later(cache);loss_before=value.item();value.backward()
    gradient=adapter.b.grad.detach().float().clone()
    norm=float(gradient.norm())
    if not kept:
        raise ValueError('The earlier recurrent-state gradient was not observed')
    state_norm=float(torch.stack(kept).sum(0).norm())
    key_width=model.config.linear_num_key_heads*model.config.linear_key_head_dim
    qkv_norms={'q':float(gradient[:,:key_width].norm()),
               'k':float(gradient[:,key_width:2*key_width].norm()),
               'v':float(gradient[:,2*key_width:].norm())}
    if qkv_norms['q'] != 0 or min(qkv_norms['k'],qkv_norms['v']) <= 0:
        raise ValueError('The isolated recurrent writer has unexpected Q/K/V credit')
    if not torch.isfinite(gradient).all() or norm<=0 or state_norm<=0:raise ValueError('Later loss did not credit the earlier recurrent write')
    record('causal-credit',cache='transformers.DynamicCache',qkv_gradient_norms=qkv_norms,loss=loss_before,adapter_gradient_norm=norm,prefix_recurrent_gradient_norm=state_norm,other_prefix_paths_detached=True,later_adapter_parameters_detached=True)
    model.zero_grad(set_to_none=True);del cache,kept,value,logits;gc.collect()
    adapter.detach_parameters=False
    control=prefix_cache(DynamicCache);isolate(control,False);adapter.detach_parameters=True
    detached,control_logits=later(control)
    if detached.requires_grad or detached.item()!=loss_before:raise ValueError('Detached causal control differs')
    record('detached-credit-control',same_forward_loss=True,requires_grad=False)
    del control,detached,control_logits;gc.collect()
    with torch.no_grad():adapter.b.copy_((adapter.b.float()-.001*gradient).to(torch.bfloat16))
    changed=int(torch.count_nonzero(adapter.b));adapter.detach_parameters=False
    if changed==0:raise ValueError('No recurrent-write parameter changed')
    save_file({'a':adapter.a.detach().contiguous(),'b':adapter.b.detach().contiguous()},str(args.out/'adapter.safetensors'))
    basis=live_basis(file_digest(args.out/'adapter.safetensors'))
    record('updated',changed_elements=changed,basis=basis)
    with torch.no_grad():
        cache=prefix_cache(DynamicCache)
        state=export_hybrid_state(cache,model.config,basis=basis)
        value,logits=later(restore_hybrid_state(state,model.config,basis=basis))
        probe_loss=value.item();probe_logits=logits.clone()
        cache=restore_hybrid_state(state,model.config,basis=basis)
        suffix=torch.tensor([ids[cut:]])
        h=model.model(suffix,past_key_values=cache,use_cache=True).last_hidden_state
        generated=[]
        for _ in range(32):
            token=int(model.lm_head(h[:,-1:,:]).float()[0,0].argmax())
            generated.append(token)
            if token==tok.eos_token_id:break
            h=model.model(torch.tensor([[token]]),past_key_values=cache,use_cache=True).last_hidden_state
    text=tok.decode(generated,skip_special_tokens=True)
    if not generated or generated[-1]!=tok.eos_token_id:raise ValueError('Generated document did not finish in its declared budget')
    memory=Memory(args.out/'state');memory.initialize(DEFAULT_IDENTITY)
    source=memory.put_source('notice','\n'.join(episode['observations']))
    memory.set_task('reply',episode['question'],filenames=['answer.md'])
    snapshot=memory.snapshot('reply',[source],basis)
    proposal=memory.record_proposal(snapshot,{'filename':'answer.md','content':text+' [notice@1]\n','citations':['notice@1']},metrics={'candidate':file_digest(args.out/'adapter.safetensors'),'training_experience':episode['id']})
    receipt=memory.commit(proposal['id'],core_identity=basis)
    actual=(memory.root/receipt['path']).read_text()
    if actual!=text+' [notice@1]\n':raise ValueError('Published generated text differs')
    payload={'cache':state,'loss':probe_loss,'logits':probe_logits,'source_snapshot':snapshot,'receipt':receipt}
    torch.save(payload,args.out/'hybrid-state.pt')
    result.update(status='completed',admitted=False,generated=text,expected=answer,exact_task_match=text.strip()==answer,loss_before=loss_before,loss_after=probe_loss)
    record('candidate-state-and-artifact',loss=probe_loss,generated=text,expected=answer,exact_task_match=text.strip()==answer,receipt=receipt,state_bytes=(args.out/'hybrid-state.pt').stat().st_size)



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'experience', 'collection-state', 'out', 'source-before', 'preflight'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--reload-only', action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    result_path=args.out/('reload-result.json' if args.reload_only else 'model-result.json')
    if result_path.exists() or (not args.reload_only and any((args.out/name).exists()
            for name in ('adapter.safetensors','hybrid-state.pt','state'))):
        raise ValueError('Use a fresh result directory; retained evidence is preserved')
    result={'schema':'nanoai.recurrent-credit-result.v2','status':'running','phases':[],
            'scope':'causal credit and complete-state integration; no quality-gain or admission claim',
            'source':str(args.source.resolve()),'experience_file':str(args.experience.resolve()),
            'candidate':str((args.out/'adapter.safetensors').resolve())}
    started=time.monotonic()

    def record(phase, **data):
        row={'phase':phase,'seconds':time.monotonic()-started,**data}
        result['phases'].append(row)
        result_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)

    try:
        run(args,result,record)
    except BaseException as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        record('failure',error=str(error))
        raise


if __name__ == '__main__':
    main()
