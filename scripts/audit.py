"""Audit actual saved models, labels and every Test row; report all final policies."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml
from hh_eval_utils import atomic_json,identity,checkpoint_identity,paired_summary
from infer import load_rows,evaluation_rows,validate_batch


def training(root,name,smoke):
    from safetensors import safe_open
    reward=name in ['golden_rm','gemma_rm']
    directory=root/'models'/name;checkpoint=directory/'last_checkpoint' if reward else directory
    minimum=6e9 if name=='golden_rm' else 2e9 if reward else 2.5e9
    files=list(checkpoint.glob('model*.safetensors'));parameters=0
    for p in files:
        with safe_open(p,framework='pt',device='cpu') as reader:
            parameters+=sum(math.prod(reader.get_slice(k).get_shape()) for k in reader.keys())
    if parameters<minimum: raise ValueError(f'Not full-size: {name} {parameters}')
    state=json.loads((directory/'trainer_state.json').read_text())
    expected=2 if smoke else 183 if name=='golden_rm' else 368 if reward else 736
    if state['global_step']!=expected or (not smoke and state['epoch']!=1): raise ValueError('Incomplete training')
    gradients=[r['grad_norm'] for r in state['log_history'] if 'grad_norm' in r]
    if not gradients or not all(math.isfinite(g) and g>0 for g in gradients): raise ValueError('Missing/invalid gradients')
    metrics=json.loads((directory/'all_results.json').read_text())
    if not all(math.isfinite(metrics[k]) for k in ['train_loss','eval_loss']): raise ValueError('Invalid losses')
    if reward:
        from transformers import AutoTokenizer,AutoConfig
        from dp3o_alignment.reward_utils import validate_reward_padding
        tok=AutoTokenizer.from_pretrained(checkpoint);cfg=AutoConfig.from_pretrained(checkpoint)
        validate_reward_padding(tok,cfg)
        if tok.model_max_length!=4096 or tok.padding_side!='right' or tok.truncation_side!='left':
            raise ValueError('Reward tokenization protocol changed')
    if name=='sft':
        eos=json.loads((directory/'sft_label_audit.json').read_text())
        for split in ['train','val']:
            if eos[split]['real_eos_targets']!=eos[split]['supervised_eos_targets']:
                raise ValueError('SFT real EOS labels were masked')
    result=dict(name=name,stored_parameters=parameters,steps=state['global_step'],
                train_loss=metrics['train_loss'],eval_loss=metrics['eval_loss'],checkpoint_identity=checkpoint_identity(checkpoint))
    atomic_json(root/f'audits/{name}.json',result)
    return result


def summarize(scored,bootstrap_samples=2000):
    rows=scored['sft'];ids=[r['row_id'] for r in rows];prompts=[r['prompt_id'] for r in rows]
    if not ids or len(set(ids))!=len(ids): raise ValueError('Empty or duplicated Test IDs')
    for other in scored.values():
        if ([r['row_id'] for r in other]!=ids or [r['prompt_id'] for r in other]!=prompts
                or not all(math.isfinite(r['reward']) for r in other)): raise ValueError('Mismatched Test rows')
    groups,inverse=np.unique(prompts,return_inverse=True)
    draws=np.random.default_rng(42).integers(0,len(groups),size=(bootstrap_samples,len(groups)),dtype=np.int32)
    models={}
    for name,values in scored.items():
        rewards=[r['reward'] for r in values]
        models[name]=dict(mean_reward=float(np.mean(rewards)),comparisons={base:paired_summary(rewards,
            [r['reward'] for r in scored[base]],draws,inverse,len(groups)) for base in ['sft','dpo']},
            mean_generated_tokens=float(np.mean([r['generated_tokens'] for r in values])),
            length_limit_pct=100*float(np.mean([r['hit_length_limit'] for r in values])),
            empty_pct=100*float(np.mean([not r['response'].strip() for r in values])))
    return dict(rows=len(rows),unique_prompts=len(groups),models=models,
                bootstrap_samples=bootstrap_samples,bootstrap_seed=42,bootstrap_unit='normalized Test prompt cluster')


def verify_inference(path,mode,source,model,input_rows,cfg):
    manifest=json.loads(Path(str(path)+'.manifest.json').read_text())
    if (manifest['mode']!=mode or manifest['data']!=str(source) or manifest['model']!=str(model)
            or manifest['model_identity']!=checkpoint_identity(model) or manifest['input_hash']!=identity(input_rows)
            or manifest['workers']!=4): raise ValueError('Inference provenance mismatch')
    if mode in ['generate','score']:
        for k in ['seed','temperature','max_new_tokens','max_prompt_tokens']:
            if manifest[k]!=cfg['evaluation'][k]: raise ValueError('Changed evaluation protocol')
        size=cfg['evaluation']['generation_batch_size' if mode=='generate' else 'scoring_batch_size']
        if manifest['batch_size']!=size: raise ValueError('Changed evaluation batch size')
    elif manifest['batch_size']!=1: raise ValueError('Pair annotation must use batch size 1')


def report(root,profile,smoke):
    from annotate_bt import sample_rows
    from run_pipeline import plan
    cfg=yaml.safe_load(profile.read_text());names=['sft','dpo']+[f'dp3o_{a:.1f}' for a in cfg['alphas']]
    for step in plan(root,profile,smoke):
        if step.config:
            saved=yaml.safe_load((root/f'configs/{step.name}.yaml').read_text())
            saved.pop('resume_from_checkpoint',None)
            if saved!=step.config: raise ValueError('Saved training config differs from plan')
    # Recompute BT labels, check orientation and proxy metadata, not merely counts.
    manifest=json.loads((root/'data/manifest.json').read_text())
    for split,seed in [('train',cfg['bt_train_seed']),('val',cfg['bt_val_seed'])]:
        original=load_rows(root/f'data/pref_{split}');gold=load_rows(root/f'scores/golden_{split}.json')
        verify_inference(root/f'scores/golden_{split}.json','golden',root/f'data/pref_{split}',root/'models/golden_rm/last_checkpoint',original,cfg)
        rows,_=sample_rows(original,gold,seed,cfg['bt_scale'])
        if load_rows(root/f'bt/pref_{split}.json')!=rows: raise ValueError('BT labels changed')
        proxy=load_rows(root/f'scores/proxy_{split}.json');validate_batch(proxy,rows,'proxy')
        verify_inference(root/f'scores/proxy_{split}.json','proxy',root/f'bt/pref_{split}.json',root/'models/gemma_rm/last_checkpoint',rows,cfg)
    test=load_rows(root/'data/pref_test');expected=evaluation_rows(test)
    if len(test)!=(8 if smoke else 2548): raise ValueError('Incomplete Test set')
    if identity([identity(r) for r in test])!=manifest['split_hashes']['test']: raise ValueError('Test data changed')
    audits={name:training(root,name,smoke) for name in ['golden_rm','gemma_rm',*names]}
    scored={}
    for name in names:
        folder=root/'evaluation/test';generation=folder/f'{name}.generations.json';scores=folder/f'{name}.scores.json'
        generated=load_rows(generation);rows=load_rows(scores)
        validate_batch(generated,expected,'generate');validate_batch(rows,generated,'score')
        verify_inference(generation,'generate',root/'data/pref_test',root/'models'/name,expected,cfg)
        verify_inference(scores,'score',root/'data/pref_test',root/'models/golden_rm/last_checkpoint',generated,cfg)
        scored[name]=rows
    result=summarize(scored,cfg['evaluation']['bootstrap_samples'])
    result.update(scale=1.,seed=cfg['seed'],smoke=smoke,training=audits,
        limitations=['Single training seed; bootstrap is across Test prompts, not training seeds.',
                     'Golden RM is both BT simulator and judge, not independent human evaluation.',
                     'Original prompt overlaps retained. All requested final checkpoints reported.',
                     'Smoke metrics only validate execution and are not benchmark results.' if smoke else
                     'Reference protocol was explored on Test; this is not an untouched confirmatory test.'])
    lines=['# '+('SMOKE VALIDATION ONLY' if smoke else 'HH BT scale=1.0, Pythia 2.8B + Gemma RM'),'',
           f'{len(test)} Test rows. Strict wins / all rows; ties not wins.','',
           '| Model | Mean reward | Delta vs DPO | Win vs SFT (%) | Win vs DPO (%) |',
           '| --- | ---: | ---: | ---: | ---: |']
    for name,v in result['models'].items():
        c=v['comparisons'];ws='—' if name=='sft' else f"{c['sft']['win_pct']['value']:.2f}"
        wd='—' if name=='dpo' else f"{c['dpo']['win_pct']['value']:.2f}"
        lines.append(f"| {name} | {v['mean_reward']:.4f} | {c['dpo']['delta']['value']:+.4f} | {ws} | {wd} |")
    lines+=['',*['- '+v for v in result['limitations']]]
    atomic_json(root/'evaluation/test/table.json',result)
    (root/'evaluation/test/table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['training','report']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name');p.add_argument('--profile',type=Path);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.mode=='training': training(a.root,a.name,a.smoke)
    else: report(a.root,a.profile,a.smoke)
