"""Full-source bidirectional meaning learning, actual artifacts and recovery."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import random
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def messages(row, *, examples=()):
    text=row['question']+'\n'+'\n'.join(row['observations'])
    if examples:
        text+='\nVerified examples:\n'+'\n'.join('\n'.join(e['observations'])+'\nResponse: '+e['target'] for e in examples)
    return [{'role':'user','content':text}]


def assess(text, criteria):
    from nanoai.meaning import assess as semantic_assess
    if 'direction' in criteria:return semantic_assess(text,criteria)
    value=text.strip().casefold().removesuffix('.')
    if criteria['kind']=='exact':return {'task_content':value in criteria['accepted']}
    return {'required_content':all(x in value for x in criteria['required']),
            'contradiction_absent':all(x not in value for x in criteria['forbidden'])}


def assessed_artifact(memory,row,arm,generation,prompt,*,source_basis,corpus_identity,evaluator):
    """Use actual source records at the journal's snapshot boundary."""
    from nanoai.experience import Experience
    task=arm+'-'+row['id'];source_id='source-'+task
    source=memory.put_source(source_id,json.dumps({'observations':row['observations']},ensure_ascii=False))
    memory.set_task(task,row['question'],filenames=['answer.txt'])
    snapshot=memory.snapshot(task,[source],source_basis+'@'+arm)
    owner=Experience(memory)
    experience=owner.record_generation(snapshot,question=row['question'],observations=row['observations'],messages=prompt,
        generation=generation,representation_identity=corpus_identity,source_id=source_id,
        metadata={'case_id':row['id'],'arm':arm,'direction':row['direction'],'family':row['family'],'corpus':corpus_identity})
    receipt=owner.execute(experience['id'])
    rubric={'scope':'Declared finite meaning and actual generated artifact; original English annotation scope retained',
            'criteria':row['criteria'],'training_target':row['target'] if row['split']=='train' else None}
    assessment=owner.assess_generation(experience['id'],rubric=rubric,evaluator_identity=evaluator,evaluate=assess)
    return {'id':row['id'],'arm':arm,'direction':row['direction'],'family':row['family'],'split':row['split'],
            'generated':generation,'checks':assessment['checks'],'correct':assessment['observed_outcome']=='yes',
            'experience_id':experience['id'],'receipt':receipt}


def observe_counterexample(memory,first,second,symbols,*,task,source_basis):
    from nanoai.meaning import interpretations,separate_meanings
    from nanoai.steps import Steps
    witness=separate_meanings(first,second,symbols)
    if witness is None:return None
    left,right=interpretations(first,symbols),interpretations(second,symbols)
    readings=[*left,*right];traces=[[] for _ in readings];observed=[]
    memory.set_task(task,'Compare the interpretations through linked separating observations.',filenames=['inquiry.txt'])
    variables=sorted(symbols);steps=Steps(memory)
    for index,assignment in enumerate(witness['assignments']):
        source=memory.put_source(task+'-case-'+str(index),json.dumps({'type':'finite_relation_v1','variables':variables,
            'worlds':[[assignment[v] for v in variables]],'scope':'One separating assignment; interpretation identity persists across cases'}))
        snapshot=memory.snapshot(task,[source],source_basis)
        def perform(op,arguments):
            proposal=steps.propose(snapshot,op,arguments);return proposal['id'],steps.execute(proposal['id'])
        read,_=perform('source.read',{'source':source['id']});decoded,_=perform('relation.decode',{'step':read})
        for rid,expression in enumerate(readings):
            _,value=perform('relation.query',{'step':decoded,'expression':expression})
            possible=value['result']['value']['possible_values']
            if len(possible)!=1:raise ValueError('A singleton operation did not determine its output')
            traces[rid].append(possible[0]);observed.append(value)
    actual_left=sorted(set(map(tuple,traces[:len(left)])));actual_right=sorted(set(map(tuple,traces[len(left):])))
    if actual_left!=witness['first_traces'] or actual_right!=witness['second_traces'] or actual_left==actual_right:
        raise ValueError('Actual operation traces differ from the separating semantic witness')
    return {'witness':witness,'observations':observed,'actual_first_traces':actual_left,'actual_second_traces':actual_right,
            'scope':'Finite simulated interpretation testing; natural-language intent remains the annotated source premise'}


def run(args,result,record):
    import torch
    from transformers import DynamicCache
    from nanoai.agent import DEFAULT_IDENTITY
    from nanoai.core import digest_bytes,file_digest
    from nanoai.memory import Memory,StaleState,canonical
    from nanoai.experience import Experience
    from nanoai.meaning import interpretations
    from nanoai.steps import Steps,strict_json
    from nanoai.torch_source import load_full_text,export_hybrid_state,restore_hybrid_state
    from nanoai.generative_learning import (attach,basis,encode,cache_prefix,loss,chunked_backward,
        candidate_bytes,restore_candidate,generate,RECIPE)

    protocol=json.loads(args.protocol.read_text());corpus=json.loads(args.corpus.read_text())
    if file_digest(args.corpus)!=protocol['corpus_digest']:
        raise ValueError('The frozen corpus changed')
    if digest_bytes(canonical({k:v for k,v in corpus.items() if k!='identity'}))!=corpus['identity']:
        raise ValueError('Corpus content identity differs')
    for relative,expected in protocol['implementation'].items():
        if file_digest(Path(__file__).resolve().parents[1]/relative)!=expected:
            raise ValueError('A frozen experiment implementation changed: '+relative)
    preflight=json.loads(args.preflight.read_text())
    if (preflight.get('model')!='Qwen3.5-2B-BF16' or preflight.get('thinking') is not True
            or preflight.get('reasoning_token_limit',0) is not None or not preflight.get('parent_source')):
        raise ValueError('Sourced native launch preflight is required')
    if protocol.get('max_tokens') is not None:
        raise ValueError('Active meaning evaluation requires natural generation without a token quota')
    if any(not r.get('teaching_reason') for r in corpus['rows'] if r['split'] in {'train','replay','validation'}):
        raise ValueError('Thinking-enabled learning needs the source-attributed rationale corpus')
    manifest=json.loads(args.source_before.read_text())
    if manifest!={p.name:{'bytes':p.stat().st_size,'digest':file_digest(p)} for p in sorted(args.source.iterdir()) if p.is_file()}:
        raise ValueError('Original model source manifest changed')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(protocol['seed'])
    model,tok=load_full_text(args.source)
    if protocol.get('source_residency') == 'read-only-source-madvise-dontneed-v1':
        from nanoai.source_residency import SourceResidency
        SourceResidency(args.source).install(model)
    elif protocol.get('source_residency') is not None:
        raise ValueError('Unqualified source residency schedule')
    parameters=attach(model)
    source_basis=basis(manifest,model)
    initial={n:p.detach().clone() for n,p in parameters.items()}
    memory=Memory(args.out/'state');memory.initialize(DEFAULT_IDENTITY);owner=Experience(memory)

    if args.reload_only:
        saved=torch.load(args.out/'continuation.pt',map_location='cpu',weights_only=True)
        data=(args.out/'selected.safetensors').read_bytes();identity=digest_bytes(data)
        restore_candidate(parameters,data,expected_identity=identity,expected_basis=source_basis)
        continuation_basis=digest_bytes(canonical({'source_basis':source_basis,'candidate':identity,'tokens':saved['prompt_ids'],'corpus':corpus['identity']}))
        with memory.locked():memory._validate_snapshot(saved['snapshot'])
        cache=restore_hybrid_state(saved['cache'],model.config,basis=continuation_basis)
        with torch.no_grad():
            hidden=model.model(torch.tensor([saved['prompt_ids'][-1:]]),past_key_values=cache,use_cache=True).last_hidden_state
            logits=model.lm_head(hidden[:,-1:,:]).float()
        if not torch.equal(logits,saved['logits']):raise ValueError('Fresh candidate continuation changed')
        export=owner.export(saved['experience_id'])
        if export['eligibility']['status']!='checked':raise ValueError('Fresh experience is not eligible')
        receipt=owner.execute(saved['experience_id']);path=memory.root/receipt['path']
        before=(path.read_bytes(),path.stat().st_ino,path.stat().st_mtime_ns)
        memory.recover();owner.execute(saved['experience_id'])
        if before!=(path.read_bytes(),path.stat().st_ino,path.stat().st_mtime_ns):raise ValueError('Recovery repeated the artifact effect')
        rejections=[]
        for label,action in [
            ('candidate_basis',lambda:restore_candidate(parameters,data,expected_identity=identity,expected_basis='changed')),
            ('candidate_bytes',lambda:restore_candidate(parameters,data[:-1]+bytes([data[-1]^1]),expected_identity=identity,expected_basis=source_basis)),
            ('hybrid_basis',lambda:restore_hybrid_state(saved['cache'],model.config,basis='changed'))]:
            try:action()
            except ValueError:rejections.append(label)
            else:raise ValueError('Incompatible candidate was accepted')
        source=export['source_id'];current=memory.state['sources'][source]
        memory.put_source(source,json.dumps({'observations':['The source has been corrected.']}),expected_revision=current['revision'])
        if owner.export(saved['experience_id'])['eligibility']['status']!='invalidated':raise ValueError('Source correction left experience current')
        result.update(status='completed',exact_continuation=True,once_only=True,rejections=rejections,source_correction_invalidates=True)
        record('fresh-recovery',candidate=identity,hybrid_bytes=(args.out/'continuation.pt').stat().st_size)
        return

    train=[r for r in corpus['rows'] if r['split']=='train']
    replay=[r for r in corpus['rows'] if r['split']=='replay']
    validation=[r for r in corpus['rows'] if r['split']=='validation']
    evaluation=[r for r in corpus['rows'] if r['split'] in {'confirmation','control'}]
    record('loaded',basis=source_basis,trainable=sum(p.numel() for p in parameters.values()),recipe=RECIPE,
           training=len(train),replay=len(replay),distinct_replay=len({r['observations'][0] for r in replay}),confirmation=len(evaluation))

    def publish(row,arm,*,examples=()):
        prompt=messages(row,examples=examples);t=time.monotonic()
        generation=generate(model,tok,prompt,max_tokens=protocol['max_tokens'])
        elapsed=time.monotonic()-t
        record('generation',id=row['id'],arm=arm,generation=generation,seconds=elapsed)
        outcome=assessed_artifact(memory,row,arm,generation,prompt,source_basis=source_basis,
                                  corpus_identity=corpus['identity'],evaluator=protocol['evaluator'])
        outcome.update(seconds=elapsed,prompt_tokens=len(tok.apply_chat_template(prompt,enable_thinking=True,add_generation_prompt=True,tokenize=True,return_dict=False)))
        result.setdefault('rows',[]).append(outcome)
        record('artifact',id=row['id'],arm=arm,correct=outcome['correct'],checks=outcome['checks'],seconds=elapsed)
        return outcome

    collection_ids=protocol.get('collection_case_ids',[r['id'] for r in train if r['direction']=='english_to_math'])
    eligible={r['id'] for r in train if r['direction']=='english_to_math'}
    if (not isinstance(collection_ids,list) or not collection_ids or len(set(collection_ids))!=len(collection_ids)
            or not set(collection_ids)<=eligible):
        raise ValueError('Collection schedule must identify distinct teaching cases')
    collected={}
    # Observe the source core's own interpretations before deriving correction.
    for row in train:
        if row['id'] not in collection_ids:continue
        outcome=publish(row,'collection')
        current=owner.export(outcome['experience_id'])
        if current['eligibility']['status']!='checked':raise ValueError('Collected experience is not eligible')
        if current['assessment']['corrective_supervision']['training_target']!=row['target']:
            raise ValueError('Checked experience and frozen teaching target differ')
        collected[row['id']]=current
        if not outcome['correct']:
            try:
                predicted=strict_json(outcome['generated']['text']);interpretations(predicted,row['criteria']['symbols'])
            except (ValueError,TypeError,KeyError):predicted=None
            if predicted is not None:
                observed=observe_counterexample(memory,predicted,row['criteria']['expression'],row['criteria']['symbols'],
                    task='inquiry-'+row['id'],source_basis=source_basis)
                if observed is not None:record('counterexample-observed',case=row['id'],**observed)

    chunked_backward(model)
    examples={r['id']:encode(tok,messages(r),
        collected[r['id']]['assessment']['corrective_supervision']['training_target'] if r['id'] in collected else r['target'],
        reasoning=r['teaching_reason'])
        for r in [*train,*replay,*validation]}
    prefixes={}
    # Repeated replay samples share only source-identical complete token inputs.
    cache_by_tokens={}
    for row in [*train,*replay,*validation]:
        example=examples[row['id']];key=digest_bytes(canonical(example))
        if key not in cache_by_tokens:cache_by_tokens[key]=cache_prefix(model,example,source_basis=source_basis)
        prefixes[row['id']]=cache_by_tokens[key]
    record('prefixes',distinct=len(cache_by_tokens),semantic_tokens=sum(len(examples[r['id']]['target_ids']) for r in train),replay_tokens=sum(len(examples[r['id']]['target_ids']) for r in replay))

    @torch.no_grad()
    def losses(rows):
        return sum(float(loss(model,examples[r['id']],prefix=prefixes[r['id']],source_basis=source_basis)[0]) for r in rows)/len(rows)
    native_validation=losses(validation);native_replay=losses(replay)
    record('baseline-loss',validation=native_validation,replay=native_replay)
    masters={n:torch.nn.Parameter(p.detach().float().clone()) for n,p in parameters.items()}
    optimizer=torch.optim.AdamW(list(masters.values()),lr=protocol['learning_rate'],weight_decay=0)
    best_score=float('inf');best_data=None;best_epoch=None
    for epoch in range(1,protocol['epochs']+1):
        order=[*train,*replay];random.Random(protocol['seed']+epoch).shuffle(order)
        total=0
        for row in order:
            model.zero_grad(set_to_none=True);optimizer.zero_grad(set_to_none=True)
            value,_=loss(model,examples[row['id']],prefix=prefixes[row['id']],source_basis=source_basis)
            if not bool(torch.isfinite(value)):raise ValueError('Nonfinite candidate loss')
            total+=float(value.detach());value.backward()
            for name,p in parameters.items():
                if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise ValueError('Invalid candidate gradient')
                masters[name].grad=p.grad.float().clone()
            torch.nn.utils.clip_grad_norm_(list(masters.values()),1.0);optimizer.step()
            with torch.no_grad():
                for name,p in parameters.items():p.copy_(masters[name].to(p.dtype))
            del value
        val=losses(validation);ret=losses(replay);score=val+protocol['replay_selection_weight']*ret
        data=candidate_bytes(parameters,source_basis=source_basis)
        (args.out/f'epoch-{epoch}.safetensors').write_bytes(data)
        if score<best_score:best_score,best_data,best_epoch=score,data,epoch
        record('epoch',epoch=epoch,training_loss=total/len(order),validation=val,replay=ret,selection_score=score)
    (args.out/'selected.safetensors').write_bytes(best_data)
    candidate=owner.register_candidate(best_data,feature_basis=source_basis,tensor_format='safetensors',training={'recipe':RECIPE,'corpus':corpus['identity'],
        'epoch':best_epoch,'selection_score':best_score,'observed_training_experiences':len(collected),
        'experience_ids':[r['id'] for r in collected.values()],
        'supervision':'Checked generated experience corrections plus exact authored bidirectional pairs and separate replay; full data budget recorded'})
    record('selected',candidate=candidate['id'],epoch=best_epoch,selection='Validation and teaching replay only; confirmation unread for selection')
    model.zero_grad(set_to_none=True);del masters,optimizer,prefixes,cache_by_tokens;gc.collect()
    for arm in ('original','retrieved','learned'):
        if arm=='learned':restore_candidate(parameters,best_data,expected_identity=candidate['id'],expected_basis=source_basis)
        else:
            with torch.no_grad():
                for n,p in parameters.items():p.copy_(initial[n])
        for row in evaluation:
            examples_for_prompt=[]
            if arm=='retrieved' and row['split']=='confirmation':
                eligible=[r for r in train if r['direction']==row['direction']]
                # A fixed pair uses only training examples and no test labels.
                examples_for_prompt=[eligible[2],eligible[4]]
            publish(row,arm,examples=examples_for_prompt)
    # The model is now the selected candidate. Save a complete hybrid boundary
    # and the observed artifact for independent fresh-process recovery.
    probe=evaluation[0];prompt=messages(probe)
    ids=tok.apply_chat_template(prompt,enable_thinking=True,add_generation_prompt=True,tokenize=True,return_dict=False)
    continuation_basis=digest_bytes(canonical({'source_basis':source_basis,'candidate':candidate['id'],'tokens':ids,'corpus':corpus['identity']}))
    with torch.no_grad():
        cache=DynamicCache(config=model.config)
        model.model(torch.tensor([ids[:-1]]),past_key_values=cache,use_cache=True)
        hybrid=export_hybrid_state(cache,model.config,basis=continuation_basis)
        h=model.model(torch.tensor([ids[-1:]]),past_key_values=cache,use_cache=True).last_hidden_state
        logits=model.lm_head(h[:,-1:,:]).float().clone()
    row=next(r for r in result['rows'] if r['arm']=='learned' and r['id']==probe['id'])
    experience=owner.export(row['experience_id'])
    torch.save({'cache':hybrid,'logits':logits,'prompt_ids':ids,'snapshot':experience['snapshot'],
                'experience_id':experience['id']},args.out/'continuation.pt')
    scores={}
    for arm in ('original','retrieved','learned'):
        selected=[r for r in result['rows'] if r['arm']==arm]
        scores[arm]={'confirmation':sum(r['correct'] for r in selected if r['split']=='confirmation'),
                     'control':sum(r['correct'] for r in selected if r['split']=='control'),
                     'seconds':sum(r['seconds'] for r in selected)}
    result.update(status='completed',scores=scores,candidate=candidate['id'],admitted=False)
    record('completed',scores=scores,candidate=candidate['id'],admitted=False,
           scope='One seed, four held-out formula families in both directions, four retention/scope controls; no broad capability or admission claim')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('out','source','preflight','source-before','protocol','corpus'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reload-only',action='store_true');args=parser.parse_args()
    output=args.out/('reload-result.json' if args.reload_only else 'model-result.json')
    if output.exists():raise ValueError('Use a fresh result path and retain prior evidence')
    result={'schema':'nanoai.meaning-learning-result.v1','status':'running','phases':[]};started=time.monotonic()
    def record(phase,**data):
        row={'phase':phase,'elapsed_seconds':time.monotonic()-started,**data};result['phases'].append(row)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(row,ensure_ascii=False),flush=True)
    try:run(args,result,record)
    except BaseException as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc());record('failure',error=str(error));raise


if __name__=='__main__':main()
