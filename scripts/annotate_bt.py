"""Sample fixed BT labels once, preserving raw scores and original row identity."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from hh_eval_utils import atomic_json, identity
from bootstrap import external_path


def scalar(value):
    if isinstance(value,(list,tuple)):
        if len(value)!=1: raise ValueError('Expected one scalar reward')
        value=value[0]
    value=float(value)
    if not math.isfinite(value): raise ValueError('Nonfinite reward')
    return value


def sample_rows(original, scored, seed, scale=1.):
    if not math.isfinite(scale) or scale<=0 or len(original)!=len(scored) or not len(original):
        raise ValueError('Invalid scale or pair counts')
    rng=np.random.Generator(np.random.PCG64(seed));rows=[];probs=[]
    for i,(before,score) in enumerate(zip(original,scored)):
        if any(before[k]!=score[k] for k in ('chosen','rejected')):
            raise ValueError(f'Score/pair mismatch at row {i}')
        if before['chosen'][:-1]!=before['rejected'][:-1]: raise ValueError('Pair prompt mismatch')
        a,b=scalar(score['chosen_rewards']),scalar(score['rejected_rewards'])
        p=float(np.exp(-np.logaddexp(0.,-scale*(a-b))));u=float(rng.random());flip=u>=p
        row=dict(before,chosen=before['rejected'] if flip else before['chosen'],
                 rejected=before['chosen'] if flip else before['rejected'],
                 chosen_rewards=[b if flip else a],rejected_rewards=[a if flip else b],
                 source_row_id=identity(before),source_index=i,bt_flip=flip,
                 bt_probability_original_chosen=p,bt_uniform=u)
        rows.append(row);probs.append(p)
    gaps=np.array([r['chosen_rewards'][0]-r['rejected_rewards'][0] for r in rows]);nonzero=gaps!=0
    return rows,dict(rows=len(rows),seed=seed,scale=scale,ordered_hash=identity(rows),
        flipped=sum(r['bt_flip'] for r in rows),flip_pct=100*np.mean([r['bt_flip'] for r in rows]),
        golden_ties=int((~nonzero).sum()),
        realized_error_vs_golden_excluding_ties_pct=100*float((gaps[nonzero]<0).mean()) if nonzero.any() else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scale',type=float,default=1.)
    p.add_argument('--train-seed',type=int,default=42)
    p.add_argument('--val-seed',type=int,default=43)
    a=p.parse_args();external_path(a.output,'annotation output')
    if a.output.exists(): raise ValueError('Refusing to overwrite labels')
    manifest=dict(scale=a.scale,temperature=1/a.scale,rule='keep original chosen iff u < sigmoid(scale * Golden gap)',
                  rng='numpy.PCG64',note='Raw rewards unscaled; Test and original SFT labels unchanged.')
    for split,seed in [('train',a.train_seed),('val',a.val_seed)]:
        original=load_from_disk(str(a.data/f'pref_{split}'))
        scored=json.loads((a.scores/f'golden_{split}.json').read_text())
        rows,stats=sample_rows(original,scored,seed,a.scale)
        atomic_json(a.output/f'pref_{split}.json',rows)
        manifest[split]=dict(stats,source_scores_hash=identity(scored))
    atomic_json(a.output/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__': main()
