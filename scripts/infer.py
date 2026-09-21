"""Independent GPU workers with atomic batch files; no NCCL evaluation collectives."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from hh_eval_utils import atomic_json, identity, checkpoint_identity, prompt_identity, invalid_model_tokens
from bootstrap import external_path


def load_rows(path):
    from datasets import load_from_disk
    if path.is_dir(): return list(load_from_disk(str(path)))
    return json.loads(path.read_text())


def evaluation_rows(data):
    return [dict(index=i,row_id=identity(row),prompt_id=prompt_identity(row['chosen'][:-1]),
                 messages=row['chosen'][:-1],chosen=row['chosen'][-1]['content']) for i,row in enumerate(data)]


def batches(rows,rank,workers,size):
    start=len(rows)*rank//workers;end=len(rows)*(rank+1)//workers
    for offset in range(start,end,size): yield offset,rows[offset:min(end,offset+size)]


def batch_path(args,rank,offset):
    return Path(str(args.output)+'.parts')/f'rank_{rank}'/f'batch_{offset}.json'


def validate_batch(saved,before,mode):
    if len(saved)!=len(before): raise ValueError('Partial batch count mismatch')
    for got,original in zip(saved,before):
        if any(got.get(k)!=v for k,v in original.items()): raise ValueError('Changed partial input/metadata')
        if mode=='generate':
            if not isinstance(got.get('response'),str) or not isinstance(got.get('generated_tokens'),int):
                raise ValueError('Invalid generation')
        elif mode=='score':
            if not math.isfinite(got.get('reward',float('nan'))): raise ValueError('Invalid reward')
        else:
            for prefix in ([''] if mode=='golden' else ['proxy_']):
                for side in ['chosen','rejected']:
                    val=got.get(prefix+side+'_rewards')
                    if not isinstance(val,list) or len(val)!=1 or not math.isfinite(val[0]):
                        raise ValueError('Invalid pair reward')


def worker(args,rows):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, set_seed
    from dp3o_alignment.data import maybe_insert_system_message
    from dp3o_alignment.reward_utils import render_reward_messages,validate_reward_padding
    device=f'cuda:{args.rank}';torch.cuda.set_device(args.rank)
    pending=[]
    for offset,batch in batches(rows,args.rank,args.workers,args.batch_size):
        path=batch_path(args,args.rank,offset)
        if path.exists(): validate_batch(json.loads(path.read_text()),batch,args.mode)
        else: pending.append((offset,batch))
    if not pending:
        print(f'rank {args.rank}: all batches already saved',flush=True);return
    tok=AutoTokenizer.from_pretrained(args.model)
    cls=AutoModelForCausalLM if args.mode=='generate' else AutoModelForSequenceClassification
    model=cls.from_pretrained(args.model,torch_dtype=torch.bfloat16).to(device).eval()
    if args.mode=='generate':
        tok.padding_side='left';tok.truncation_side='left';model.config.use_cache=True
        suppress=invalid_model_tokens(tok,model.config.vocab_size)
    else: validate_reward_padding(tok,model.config)
    stops=['<|user|>','<|assistant|>','<|system|>']
    for offset,batch in pending:
        with torch.inference_mode():
            if args.mode=='generate':
                prompts=[]
                for row in batch:
                    messages=deepcopy(row['messages']);maybe_insert_system_message(messages,tok)
                    prompts.append(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
                lengths=[len(tok(t,add_special_tokens=False)['input_ids']) for t in prompts]
                encoded=tok(prompts,padding=True,truncation=True,max_length=args.max_prompt_tokens,
                            add_special_tokens=False,return_tensors='pt').to(device)
                set_seed(args.seed+offset)
                generated=model.generate(**encoded,do_sample=True,temperature=args.temperature,top_p=1.,top_k=0,
                    max_new_tokens=args.max_new_tokens,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,
                    use_cache=True,stop_strings=stops,tokenizer=tok,suppress_tokens=suppress)
                result=[]
                for i,row in enumerate(batch):
                    tokens=generated[i,encoded['input_ids'].shape[1]:].tolist();eos=tok.eos_token_id in tokens
                    if eos: tokens=tokens[:tokens.index(tok.eos_token_id)+1]
                    text=tok.decode(tokens,skip_special_tokens=True)
                    positions=[text.find(s) for s in stops if s in text];role_stop=bool(positions)
                    if role_stop: text=text[:min(positions)]
                    response=text.strip();limit=not eos and not role_stop and len(tokens)>=args.max_new_tokens
                    result.append(dict(row,response=response,generated_tokens=len(tok(response,add_special_tokens=False)['input_ids']),
                        stop_reason='role_marker' if role_stop else 'eos' if eos else 'length_limit' if limit else 'other',
                        hit_length_limit=limit,prompt_truncated=lengths[i]>args.max_prompt_tokens))
            else:
                result=[]
                # Pair annotation uses batch size 1, matching the validated raw-logit recipe.
                sides=['chosen','rejected'] if args.mode in ['golden','proxy'] else ['response']
                values={};truncated={}
                for side in sides:
                    messages=[r[side] if side!='response' else r['messages']+[dict(role='assistant',content=r['response'])] for r in batch]
                    texts=[render_reward_messages(tok,m) for m in messages]
                    lengths=[len(tok(t)['input_ids']) for t in texts]
                    encoded=tok(texts,padding=True,truncation=True,return_tensors='pt').to(device)
                    scores=model(**encoded,use_cache=False).logits[:,0].float().cpu().tolist()
                    if not all(math.isfinite(v) for v in scores): raise ValueError('Nonfinite RM score')
                    values[side]=scores;truncated[side]=[n>tok.model_max_length for n in lengths]
                for i,row in enumerate(batch):
                    if args.mode=='score': result.append(dict(row,reward=values['response'][i],reward_truncated=truncated['response'][i]))
                    else:
                        prefix='' if args.mode=='golden' else 'proxy_'
                        result.append(dict(row,**{prefix+s+'_rewards':[values[s][i]] for s in sides}))
        validate_batch(result,batch,args.mode)
        atomic_json(batch_path(args,args.rank,offset),result)
        print(f'{args.mode}: rank {args.rank}, saved {offset}+{len(batch)}',flush=True)


def parent(args,rows):
    external_path(args.output,'inference output')
    if args.output.exists(): raise ValueError('Final output already exists; use pipeline --resume')
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='rank'}
    settings.update(input_hash=identity(rows),model_identity=checkpoint_identity(args.model))
    manifest=Path(str(args.output)+'.manifest.json')
    if manifest.exists():
        if json.loads(manifest.read_text())!=settings: raise ValueError('Resume settings or model/input changed')
    else: atomic_json(manifest,settings)
    children=[]
    try:
        for rank in range(args.workers):
            children.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--rank',str(rank)]))
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children): raise RuntimeError('Inference worker failed; partial batches retained')
            time.sleep(.5)
        if any(c.returncode!=0 for c in children): raise RuntimeError('Inference worker failed')
    finally:
        for child in children:
            if child.poll() is None: child.terminate()
        for child in children:
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill();child.wait()
    result=[]
    for rank in range(args.workers):
        for offset,batch in batches(rows,rank,args.workers,args.batch_size):
            saved=json.loads(batch_path(args,rank,offset).read_text());validate_batch(saved,batch,args.mode)
            result.extend(saved)
    if len(result)!=len(rows): raise ValueError('Missing result rows')
    atomic_json(args.output,result)
    print(f'Completed {args.mode}: {len(result)} rows.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['golden','proxy','generate','score'])
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--input',type=Path)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--rank',type=int)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--max-prompt-tokens',type=int,default=512)
    p.add_argument('--temperature',type=float,default=.7)
    a=p.parse_args()
    if a.workers<1 or a.batch_size<1: p.error('Positive worker and batch counts required')
    rows=load_rows(a.data)
    if a.mode in ['generate','score']: rows=evaluation_rows(rows)
    if a.mode=='score':
        if not a.input: p.error('score requires --input generations')
        generated=load_rows(a.input);validate_batch(generated,rows,'generate');rows=generated
    if not rows: raise ValueError('Empty inference dataset')
    if a.rank is None: parent(a,rows)
    else: worker(a,rows)


if __name__=='__main__': main()
