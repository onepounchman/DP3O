"""Reproduce the historical HH conversion and order, verified against frozen hashes."""
import argparse
import json
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset
from hh_eval_utils import atomic_json, identity, prompt_identity
from bootstrap import external_path

REPO=Path(__file__).resolve().parents[1]


def convert(row):
    chosen_text, rejected_text = row['chosen'], row['rejected']
    marker='\n\nAssistant:'
    index=chosen_text.rfind(marker)
    if index < 0:
        raise ValueError('Missing Assistant separator')
    prompt=chosen_text[:index+len(marker)]
    chosen=chosen_text[len(prompt):].replace(marker,'').strip()
    rejected=rejected_text[len(prompt):].replace(marker,'').strip()
    prompt=prompt.replace('\n\nHuman: ','').replace(marker,'').strip()
    if (chosen==rejected or len(prompt.split())>400 or chosen_text.count('Human: ')>1
            or rejected_text.count('Human: ')>1):
        return None
    return {key:[{'content':prompt,'role':'user'},{'content':answer,'role':'assistant'}]
            for key,answer in [('chosen',chosen),('rejected',rejected)]}


def prepare(output, config, smoke=False):
    external_path(output, 'data output')
    if output.exists():
        raise ValueError('Refusing to overwrite prepared data')
    cfg=yaml.safe_load(config.read_text())['dataset']
    converted={}
    for split in ['train','test']:
        source=load_dataset(cfg['name'],revision=cfg['revision'],split=split)
        converted[split]=[pair for row in source if (pair:=convert(row)) is not None]
    splits={'train':converted['train'][:47100], 'val':converted['train'][47100:48100],
            'test':converted['test']}
    expected=json.loads((REPO/'configs/data_protocol.json').read_text())
    full=dict(counts={s:len(v) for s,v in splits.items()},
              split_hashes={s:identity([identity(row) for row in v]) for s,v in splits.items()})
    prompts={s:{prompt_identity(row['chosen'][:-1]) for row in v} for s,v in splits.items()}
    full['prompt_overlaps']={f'{a}_{b}':len(prompts[a]&prompts[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    if full != expected:
        raise ValueError(f'Data protocol mismatch; refusing to silently change setup: {full}')
    if smoke:
        splits={s:v[:(16 if s=='train' else 8)] for s,v in splits.items()}
    for split,rows in splits.items():
        Dataset.from_list(rows).save_to_disk(str(output/f'pref_{split}'))
        if split!='test':
            Dataset.from_list([dict(messages=row['chosen']) for row in rows]).save_to_disk(str(output/f'sft_{split}'))
    atomic_json(output/'manifest.json',dict(source=cfg,full_protocol=full,smoke=smoke,
        counts={s:len(v) for s,v in splits.items()},
        split_hashes={s:identity([identity(row) for row in v]) for s,v in splits.items()}))
    print('Verified original split identities; saved data outside codebase.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=REPO/'configs/experiment.yaml')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();prepare(args.output,args.config,args.smoke)
