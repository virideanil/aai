"""Observe native and candidate credit, full-prefix parity and exact reload."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('out','source','preflight','source-before'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    import torch
    from nanoai.core import digest_bytes, file_digest
    from nanoai.torch_source import load_full_text
    from nanoai.generative_learning import (attach,basis,encode,cache_prefix,loss,
        chunked_backward,candidate_bytes,restore_candidate)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(62113)
    result={'status':'running','phases':[]};start=time.monotonic()
    def record(phase,**data):
        row={'phase':phase,'seconds':time.monotonic()-start,**data};result['phases'].append(row)
        (args.out/'model-result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    try:
        preflight=json.loads(args.preflight.read_text())
        if preflight.get('model')!='Qwen3.5-2B-BF16' or preflight.get('thinking') is not True or not preflight.get('parent_source'):
            raise ValueError('Sourced model launch preflight is required')
        manifest=json.loads(args.source_before.read_text())
        if manifest!={p.name:{'bytes':p.stat().st_size,'digest':file_digest(p)} for p in sorted(args.source.iterdir()) if p.is_file()}:
            raise ValueError('Source manifest differs')
        model,tok=load_full_text(args.source);parameters=attach(model)
        source_basis=basis(manifest,model)
        example=encode(tok,[{'role':'user','content':'Let x mean the door is open. Express the door is closed as JSON using the operator not and the variable x. Return only JSON.'}],'{"not":"x"}',
                       reasoning='The source defines x as open. Closed negates that proposition, so the JSON applies not to x.')
        prefix=cache_prefix(model,example,source_basis=source_basis)
        record('loaded',trainable=sum(p.numel() for p in parameters.values()),input_tokens=len(example['input_ids']),answer_tokens=len(example['target_ids']),basis=source_basis)
        def measure(cached):
            model.zero_grad(set_to_none=True);t=time.monotonic()
            value,logits=loss(model,example,prefix=prefix if cached else None,source_basis=source_basis)
            forward=time.monotonic()-t;value.backward()
            return {'loss':value.item(),'logits':logits.detach().clone(),'gradients':{n:p.grad.detach().clone() for n,p in parameters.items()},'forward_seconds':forward,'total_seconds':time.monotonic()-t}
        native=measure(False)
        record('native',**{k:v for k,v in native.items() if k not in {'logits','gradients'}})
        cached_native=measure(True)
        if not torch.equal(native['logits'],cached_native['logits']) or any(not torch.equal(g,cached_native['gradients'][n]) for n,g in native['gradients'].items()):
            raise ValueError('Native frozen-prefix computation differs')
        record('native-prefix',exact_logits=True,exact_gradients=True,total_seconds=cached_native['total_seconds'])
        del cached_native;model.zero_grad(set_to_none=True);gc.collect()
        chunked_backward(model)
        candidate=measure(False)
        if not torch.equal(native['logits'],candidate['logits']):raise ValueError('Original BF16 forward changed')
        errors={n:float((g.float()-native['gradients'][n].float()).norm())/max(float(native['gradients'][n].float().norm()),1e-30) for n,g in candidate['gradients'].items()}
        record('separate-gradient-recipe',forward_equal=True,relative_l2=errors,native_equivalence_claim=False,total_seconds=candidate['total_seconds'])
        cached=measure(True)
        if not torch.equal(candidate['logits'],cached['logits']) or any(not torch.equal(g,cached['gradients'][n]) for n,g in candidate['gradients'].items()):
            raise ValueError('Candidate frozen-prefix computation differs')
        record('candidate-prefix',exact_logits=True,exact_own_recipe_gradients=True,total_seconds=cached['total_seconds'])
        with torch.no_grad():
            for n,p in parameters.items():p.add_((-0.01*cached['gradients'][n].float()).to(p.dtype))
        data=candidate_bytes(parameters,source_basis=source_basis);identity=digest_bytes(data)
        (args.out/'adapter.safetensors').write_bytes(data)
        with torch.no_grad():
            full,full_logits=loss(model,example)
            reused,reused_logits=loss(model,example,prefix=prefix,source_basis=source_basis)
        if not torch.equal(full_logits,reused_logits):raise ValueError('Updated candidate invalidated frozen-prefix assumption')
        restore_candidate(parameters,data,expected_identity=identity,expected_basis=source_basis)
        rejected=[]
        for bad in ('changed-source','changed-token-sequence','changed-candidate'):
            try:
                if bad=='changed-source':loss(model,example,prefix=prefix,source_basis='different')
                elif bad=='changed-token-sequence':loss(model,{**example,'input_ids':example['input_ids']+[1]},prefix=prefix,source_basis=source_basis)
                else:restore_candidate(parameters,data,expected_identity=identity,expected_basis='different')
            except ValueError:rejected.append(bad)
            else:raise ValueError('An incompatible record was accepted')
        result['status']='completed'
        record('qualified',updated_full_cached_equal=True,rejections=rejected,candidate_identity=identity,scope='Arithmetic and cache compatibility only; quality remains unmeasured')
    except BaseException as e:
        result.update(status='failed',error=str(e),traceback=traceback.format_exc());record('failure',error=str(e));raise


if __name__=='__main__':main()
