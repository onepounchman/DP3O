"""Stable identities, resumable batch artifacts, and paired evaluation statistics."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def prompt_identity(messages):
    return identity(re.sub(r'\s+',' ',json.dumps(messages,sort_keys=True,ensure_ascii=False)).strip())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temporary.replace(path)


def checkpoint_identity(path):
    root = Path(path)
    return identity([(str(p.relative_to(root)),p.stat().st_size,p.stat().st_mtime_ns)
                     for p in sorted(root.glob('*')) if p.is_file() and
                     (p.suffix in {'.json','.safetensors'} or p.name == 'tokenizer.model')])


def invalid_model_tokens(tokenizer, model_vocab_size):
    # Pythia has padded model vocabulary slots without tokenizer entries.
    # They cannot be decoded and would index past StopStringCriteria's table.
    return sorted(set(range(model_vocab_size))-set(tokenizer.get_vocab().values()))


def paired_summary(scores, baseline, bootstrap, inverse, groups):
    scores, baseline = np.asarray(scores,dtype=float),np.asarray(baseline,dtype=float)
    delta = scores-baseline
    result = {}
    counts = np.bincount(inverse,minlength=groups)
    for key,values in [('delta',delta),('win_pct',(delta>0)*100.0),('tie_pct',(delta==0)*100.0),
                       ('loss_pct',(delta<0)*100.0),('half_tie_win_pct',((delta>0)+0.5*(delta==0))*100.0)]:
        totals = np.bincount(inverse,weights=values,minlength=groups)
        samples = totals[bootstrap].sum(axis=1)/counts[bootstrap].sum(axis=1)
        result[key] = {'value':float(values.mean()),'ci95':np.quantile(samples,[0.025,0.975]).tolist()}
    return result
