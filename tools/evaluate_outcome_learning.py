"""Compare chosen-outcome learning with the retained oracle-supervised recipe.

The frozen panel is a diagnostic set after inspection. This run never admits a
candidate. Original learning.py and its checkpoint identity stay unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nanoai.core import digest_bytes, file_digest, process_memory, swap_bytes
from nanoai.experience import Experience
from nanoai.learning import (Features, Learner, checked_episode, checkpoint_bytes,
                             encoded_texts, evaluate, targets)
from nanoai.memory import Memory, canonical
from evaluate_developmental import run_artifacts, save


def outcome_row(record):
    """Expose only the factual chosen action and its assessed binary success."""
    checked_episode(record)  # same current-source and actual-receipt eligibility
    ids=[c['id'] for c in record['candidates']]
    behavior=record['behavior']; probabilities=behavior['probabilities']
    if (behavior['policy'] != 'uniform-v1' or len(probabilities) != len(ids)
            or any(p != 1/len(ids) for p in probabilities)):
        raise ValueError('This comparison requires its actual full-support uniform behavior')
    return {k:record['metadata'][k] for k in ('id','family','split','mode','language')} | {
        'question':record['question'],'observations':record['observations'],
        'candidates':record['candidates'],'chosen_index':ids.index(behavior['selected_id']),
        'propensity':probabilities[ids.index(behavior['selected_id'])],
        'reward':float(record['assessment']['observed_outcome']=='yes'),
        'experience_id':record['id']}


def outcome_loss(logits, episode):
    """Unclipped importance-weighted score gradient for a fixed observed act.

    Reward is 1 for independently checked success and 0 otherwise. The fixed
    1/action-count baseline uses no target label. Propensity support belongs to
    the recorded behavior, and the ratio is detached in this score estimator.
    """
    index=episode['chosen_index']; logp=logits.log_softmax(-1)[index]
    weight=logp.detach().exp()/episode['propensity']
    return -weight*(episode['reward']-1/logits.numel())*logp


def train_candidate(features, rows, validation, test, retention, *, seed, mode,
                    no_memory=False, parent=None, epochs=300):
    torch.manual_seed(seed)
    model=Learner(no_memory=no_memory) if parent is None else deepcopy(parent)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    rng=random.Random(seed);started=time.monotonic();swap=swap_bytes()
    best=None;history=[];resources=[]
    for epoch in range(epochs+1):
        if time.monotonic()-started>300:
            raise TimeoutError('Candidate exceeded its five-minute bound')
        if epoch%5==0 or epoch==epochs:
            sample={'seconds':time.monotonic()-started,**process_memory(os.getpid()),'swap':swap_bytes()}
            resources.append(sample)
            if (sample['rss_bytes']>2*1024**3 or (sample['physical_footprint_bytes'] or 0)>2*1024**3
                    or sample['swap']-swap>512*1024**2):
                raise RuntimeError('Candidate resource guard reached')
            value=evaluate(model,features,validation);score=(value['correct'],-value['loss'])
            history.append({'epoch':epoch,'validation_correct':value['correct'],'validation_loss':value['loss']})
            if best is None or score>best[0]:best=(score,deepcopy(model.state_dict()),epoch)
        if epoch==epochs:break
        model.train();order=list(rows);rng.shuffle(order)
        for offset in range(0,len(order),24):
            batch=order[offset:offset+24];optimizer.zero_grad(set_to_none=True)
            encoded=encoded_texts(model,features,batch);losses=[]
            for episode in batch:
                logits,verdicts,_=model(episode,encoded)
                if mode=='outcome':losses.append(outcome_loss(logits,episode))
                else:
                    target,classes=targets(episode)
                    losses.append(F.cross_entropy(logits.unsqueeze(0),target)+.4*F.cross_entropy(verdicts,classes))
            loss=torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite learning objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
    # Record the fixed final epoch before restoring the validation selection.
    final=evaluate(model,features,test)
    model.load_state_dict(best[1]);model.eval()
    return model,{'seed':seed,'mode':mode,'epochs':epochs,'selected_epoch':best[2],
        'history':history,'resources':resources,'seconds':time.monotonic()-started,
        'train_ids':[r['id'] for r in rows],'validation_ids':[r['id'] for r in validation],
        'test':evaluate(model,features,test),'last_epoch_test':final,
        'retention':evaluate(model,features,retention)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','features','collection-state','original-results','out','preflight'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'results.json').exists():raise ValueError('Retain prior results; use a fresh output')
    preflight=json.loads(args.preflight.read_text())
    if (preflight.get('model')!='nanoai-developmental-f32' or preflight.get('effort')!='none'
            or not preflight.get('parent_source')):raise ValueError('Sourced launch preflight required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    corpus=json.loads(args.corpus.read_text());features=Features(args.features)
    if (digest_bytes(canonical({k:v for k,v in corpus.items() if k!='identity'}))!=corpus['identity']
            or features.manifest['corpus']!=file_digest(args.corpus)):raise ValueError('Frozen corpus differs')
    memory=Memory(args.collection_state);owner=Experience(memory)
    records=[owner.export(r['id']) for r in memory.state['experiences'].values()]
    episodes=[checked_episode(r) for r in records]
    train=[e for e in episodes if e['split']=='train' and e['mode']!='correction_back']
    corrective=[e for e in episodes if e['split']=='train' and e['mode']=='correction_back']
    validation=[e for e in episodes if e['split']=='validation'];test=[e for e in episodes if e['split']=='test']
    if len(train)!=42 or len(validation)!=16 or len(test)!=32:raise ValueError('Frozen split differs')
    outcomes=[outcome_row(r) for r in records if r['metadata']['split']=='train' and r['metadata']['mode']!='correction_back']
    if any('rubric' in e or 'replayable_targets' in e for e in outcomes):raise ValueError('Outcome inputs include oracle labels')
    old=json.loads(args.original_results.read_text())
    result={'schema':'nanoai.outcome-learning.v1','status':'running','admitted':False,
        'scope':'Authored diagnostic menu, oracle gradients versus chosen-outcome score estimator; unequal supervision information',
        'corpus':corpus['identity'],'implementation':file_digest(__file__),
        'oracle_information':'target action plus five evidence labels per training episode',
        'outcome_information':'one observed binary success, chosen action and actual propensity per training episode; no evidence-head target',
        'validation_information':'same 16 validation action targets and evidence labels select every arm; no test decision enters training',
        'runs':[]}
    save(args.out/'outcome-inputs.json',outcomes)
    started=time.monotonic()
    try:
        for seed in (11,23,37):
            for no_memory in (False,True):
                name=f'seed{seed}'+('-last' if no_memory else '-recurrent')
                model,row=train_candidate(features,train,validation,test,train+validation,seed=seed,mode='oracle',no_memory=no_memory)
                prior=next(r for r in old['runs'] if r['name']==name)
                saved=torch.load(args.original_results.parent/prior['checkpoint'],weights_only=True,map_location='cpu')
                row['selected_parameters_equal_original']=all(torch.equal(v,saved['state'][k]) for k,v in model.state_dict().items())
                if not row['selected_parameters_equal_original']:raise ValueError('Oracle replay differs from the original selected candidate')
                row['name']=name;result['runs'].append(row);save(args.out/'results.json',result)
                if not no_memory:
                    revised,rr=train_candidate(features,train+corrective,validation,test,train+validation,seed=seed,mode='oracle',parent=model,epochs=100)
                    prior=next(r for r in old['runs'] if r['name']==name+'-consolidated')
                    saved=torch.load(args.original_results.parent/prior['checkpoint'],weights_only=True,map_location='cpu')
                    rr['selected_parameters_equal_original']=all(torch.equal(v,saved['state'][k]) for k,v in revised.state_dict().items())
                    if not rr['selected_parameters_equal_original']:raise ValueError('Consolidated replay differs')
                    rr['name']=name+'-consolidated';result['runs'].append(rr);save(args.out/'results.json',result)
            model,row=train_candidate(features,outcomes,validation,test,train+validation,seed=seed,mode='outcome')
            row['name']=f'seed{seed}-outcome'
            data=checkpoint_bytes(model,feature_basis=features.basis,corpus_identity=corpus['identity'],experience_ids=[e['experience_id'] for e in outcomes],training={k:v for k,v in row.items() if k not in ('test','last_epoch_test','retention')})
            (args.out/(row['name']+'.pt')).write_bytes(data)
            row['artifact_result']=run_artifacts(args.out,row['name'],corpus,features,model,data,test)
            if row['artifact_result']['correct']!=row['test']['correct']:raise ValueError('Observed artifact outcome differs')
            result['runs'].append(row);save(args.out/'results.json',result)
            print(json.dumps({'seed':seed,'outcome_correct':row['test']['correct'],'final_epoch_correct':row['last_epoch_test']['correct']}),flush=True)
        result['status']='completed'
    except BaseException as error:
        result.update(status='failed',error=str(error));raise
    finally:
        result['seconds']=time.monotonic()-started;save(args.out/'results.json',result)


if __name__=='__main__':main()
