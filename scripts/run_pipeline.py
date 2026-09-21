"""Independent HH -> Golden -> BT1 -> Gemma -> SFT/DPO/DP3O -> Test pipeline."""
import argparse
from dataclasses import dataclass, asdict, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml
from bootstrap import REPO, external_path
from hh_eval_utils import atomic_json, identity, checkpoint_identity


@dataclass
class Step:
    name: str
    command: list
    inputs: list
    outputs: list
    config: dict = field(default_factory=dict)
    launcher_config: dict = field(default_factory=dict)


def plan(root, profile, smoke=False, cpu_offload=False):
    cfg=yaml.safe_load(profile.read_text())
    if cpu_offload and not smoke: raise ValueError('CPU offload option is for smoke validation only')
    if cfg['bt_scale']!=1.: raise ValueError('This release fixes BT scale=1.0')
    if cfg['num_gpus']!=4 or int(os.environ.get('DP3O_NUM_GPUS','4'))!=4:
        raise ValueError('Reference full-weight recipe uses four GPUs')
    if not cfg['alphas'] or len(set(cfg['alphas']))!=len(cfg['alphas']) or any(a not in [.2,.4,.6] for a in cfg['alphas']):
        raise ValueError('Choose unique alphas from .2/.4/.6')
    py=sys.executable;data=root/'data';models=root/'models';scores=root/'scores';bt=root/'bt'
    def call(script,*args): return [py,str(REPO/'scripts'/script),*map(str,args)]
    def step(name,command,inputs,outputs,config=None):
        return Step(name,command,list(map(str,inputs)),list(map(str,outputs)),config or {})
    steps=[step('download',call('download.py','--config',profile,'--output',root/'downloads.json'),[],[root/'downloads.json']),
           step('prepare',call('prepare_data.py','--config',profile,'--output',data)+(['--smoke'] if smoke else []),
                [],[data/'manifest.json'])]
    def training(name,recipe,initial,train,val,alpha=None):
        reward=recipe in ['golden_rm','gemma_rm']
        c=yaml.safe_load((REPO/f'configs/{recipe}.yaml').read_text())
        if isinstance(initial,dict):
            c.update(model_name_or_path=initial['name'],model_revision=initial['revision'])
        else: c.update(model_name_or_path=str(initial))
        c.update(output_dir=str(models/name),seed=cfg['seed'],hub_model_id='dp3o-hh-'+name)
        if recipe=='sft': c.update(dataset_splits=[str(train),str(val)])
        else: c.update(train_data=str(train),val_data=str(val))
        if alpha is not None:
            c.update(alpha=alpha,ref_model=str(models/'sft'),dataset_splits=[str(train),str(val)])
        if smoke:
            c.update(max_steps=2,gradient_accumulation_steps=1,logging_steps=1,save_steps=1,
                     eval_steps=1,preprocessing_num_workers=1,save_total_limit=1)
        config_path=root/f'configs/{name}.yaml'
        script='run_reward_model.py' if reward else 'run_sft.py' if recipe=='sft' else 'run_dp3o.py'
        command=[py,'-m','accelerate.commands.launch','--config_file',str(REPO/f'configs/zero{3 if reward else 2}.yaml'),
                 '--num_processes','4','--main_process_port',str(cfg.get('main_process_port',29500)),
                 str(REPO/'scripts'/script),str(config_path)]
        inputs=[train,val,root/'downloads.json']+([] if isinstance(initial,dict) else [initial])
        output=models/name/('last_checkpoint' if reward else '')
        steps.append(step(name,command,inputs,[output/'config.json',models/name/'trainer_state.json',models/name/'all_results.json'],c))
        if cpu_offload:
            # Keep full model sizes and training code; move optimizer state to CPU
            # for smoke verification on shared GPUs. Use torch AdamW, no CPUAdam JIT.
            stage=3 if reward else 2
            ds=dict(train_micro_batch_size_per_gpu='auto',train_batch_size='auto',gradient_accumulation_steps='auto',
                    gradient_clipping='auto',bf16=dict(enabled=True),zero_force_ds_cpu_optimizer=False,
                    zero_optimization=dict(stage=stage,offload_optimizer=dict(device='cpu',pin_memory=True)))
            if reward: ds['zero_optimization']['stage3_gather_16bit_weights_on_model_save']=True
            launcher=yaml.safe_load((REPO/f'configs/zero{stage}.yaml').read_text())
            launcher.pop('mixed_precision',None)
            launcher['deepspeed_config']=dict(deepspeed_config_file=str(root/f'configs/{name}.deepspeed.json'),zero3_init_flag=False)
            steps[-1].launcher_config=dict(accelerate=launcher,deepspeed=ds)
            command[command.index('--config_file')+1]=str(root/f'configs/{name}.accelerate.yaml')
        steps.append(step('audit_'+name,call('audit.py','training','--root',root,'--name',name)+(['--smoke'] if smoke else []),
                          [output/'config.json'],[root/f'audits/{name}.json']))
    def infer(name,mode,source,model,out,inputs=(),generation=None):
        ev=cfg['evaluation'];bs=1 if mode in ['golden','proxy'] else ev['generation_batch_size'] if mode=='generate' else ev['scoring_batch_size']
        command=call('infer.py',mode,'--data',source,'--model',model,'--output',out,'--workers',4,'--batch-size',bs,
                     '--seed',ev['seed'],'--temperature',ev['temperature'],'--max-new-tokens',ev['max_new_tokens'],
                     '--max-prompt-tokens',ev['max_prompt_tokens'])
        if generation: command+=['--input',str(generation)]
        steps.append(step(name,command,[source,model,*inputs]+([generation] if generation else []),[out]))
    def evaluate(name):
        generation=root/f'evaluation/test/{name}.generations.json';out=root/f'evaluation/test/{name}.scores.json'
        infer('generate_'+name,'generate',data/'pref_test',models/name,generation,[root/f'audits/{name}.json'])
        infer('score_'+name,'score',data/'pref_test',models/'golden_rm/last_checkpoint',out,[],generation)
    training('golden_rm','golden_rm',cfg['models']['golden'],data/'pref_train',data/'pref_val')
    for split in ['train','val']:
        infer('golden_score_'+split,'golden',data/f'pref_{split}',models/'golden_rm/last_checkpoint',scores/f'golden_{split}.json',
              [root/'audits/golden_rm.json'])
    steps.append(step('annotate',call('annotate_bt.py','--data',data,'--scores',scores,'--output',bt,
        '--scale',cfg['bt_scale'],'--train-seed',cfg['bt_train_seed'],'--val-seed',cfg['bt_val_seed']),
        [scores/'golden_train.json',scores/'golden_val.json'],[bt/'manifest.json',bt/'pref_train.json',bt/'pref_val.json']))
    training('sft','sft',cfg['models']['policy'],data/'sft_train',data/'sft_val');evaluate('sft')
    training('gemma_rm','gemma_rm',cfg['models']['proxy'],bt/'pref_train.json',bt/'pref_val.json')
    for split in ['train','val']:
        infer('proxy_score_'+split,'proxy',bt/f'pref_{split}.json',models/'gemma_rm/last_checkpoint',scores/f'proxy_{split}.json',
              [root/'audits/gemma_rm.json'])
    training('dpo','policy',models/'sft',bt/'pref_train.json',bt/'pref_val.json',alpha=1.);evaluate('dpo')
    for alpha in cfg['alphas']:
        name=f'dp3o_{alpha:.1f}'
        training(name,'policy',models/'sft',scores/'proxy_train.json',scores/'proxy_val.json',alpha=alpha);evaluate(name)
    names=['sft','dpo']+[f'dp3o_{a:.1f}' for a in cfg['alphas']]
    steps.append(step('report',call('audit.py','report','--root',root,'--profile',profile)+(['--smoke'] if smoke else []),
                      [root/f'evaluation/test/{name}.scores.json' for name in names],
                      [root/'evaluation/test/table.json',root/'evaluation/test/table.md']))
    return steps


def fingerprint(path):
    path=Path(path)
    if path.is_file(): return hashlib.sha256(path.read_bytes()).hexdigest()
    # Dataset.map writes derived cache-*.arrow files beside immutable source shards.
    # Fingerprint declared source data and metadata, never those disposable caches.
    if (path/'state.json').is_file():
        state=json.loads((path/'state.json').read_text())
        names=['state.json','dataset_info.json']+[item['filename'] for item in state['_data_files']]
        files={}
        for name in names:
            source=(path/name).resolve()
            if not source.is_relative_to(path.resolve()):
                raise ValueError(f'Dataset shard escapes its directory: {name}')
            files[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        return identity(files)
    return checkpoint_identity(path)


def execute(root,steps,args):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        code=[p for folder in ['src','scripts','configs','setup'] for p in (REPO/folder).rglob('*')
              if p.is_file() and p.suffix in ['.py','.yaml','.json','.sh']]
        manifest=dict(profile=yaml.safe_load(args.config.read_text()),smoke=args.smoke,cpu_offload=getattr(args,'cpu_offload',False),
                      code={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code)})
        frozen=root/'run_manifest.json'
        if frozen.exists():
            if json.loads(frozen.read_text())!=manifest: raise ValueError('Frozen config/code changed; use a NEW run name')
        else: atomic_json(frozen,manifest)
        state_path=root/'completed.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
        (root/'logs').mkdir(exist_ok=True);(root/'configs').mkdir(exist_ok=True)
        try:
            for step in steps:
                if args.stages and step.name not in args.stages: continue
                missing=[p for p in step.inputs if not Path(p).exists()]
                if missing: raise ValueError(f'{step.name}: missing prerequisites {missing}')
                signature=identity(asdict(step));input_hashes={p:fingerprint(p) for p in step.inputs}
                prior=state.get(step.name)
                if args.resume and prior:
                    if prior['signature']!=signature or prior['inputs']!=input_hashes:
                        raise ValueError(f'Changed completed stage inputs/config: {step.name}')
                    if not all(Path(p).exists() and fingerprint(p)==prior['outputs'][p] for p in step.outputs):
                        raise ValueError(f'Missing/changed completed output: {step.name}')
                    print('Skipping completed',step.name,flush=True);continue
                intent=root/f'configs/{step.name}.intent.json'
                record=dict(signature=signature,inputs=input_hashes)
                if intent.exists():
                    if not args.resume or json.loads(intent.read_text())!=record:
                        raise ValueError('Partial stage requires --resume and unchanged inputs')
                else:
                    if any(Path(p).exists() for p in step.outputs): raise ValueError(f'Refusing untracked outputs: {step.name}')
                    atomic_json(intent,record)
                if step.config:
                    c=dict(step.config);directory=Path(c['output_dir'])
                    if directory.exists() and any(directory.iterdir()):
                        checkpoints=sorted(directory.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
                        if not args.resume or not checkpoints or not (checkpoints[-1]/'trainer_state.json').exists():
                            raise ValueError('Partial training lacks a resumable checkpoint; preserve it and use a new run')
                        c['resume_from_checkpoint']=str(checkpoints[-1])
                    (root/f'configs/{step.name}.yaml').write_text(yaml.safe_dump(c))
                    if step.launcher_config:
                        atomic_json(root/f'configs/{step.name}.deepspeed.json',step.launcher_config['deepspeed'])
                        (root/f'configs/{step.name}.accelerate.yaml').write_text(yaml.safe_dump(step.launcher_config['accelerate']))
                elif args.resume and all(Path(p).exists() for p in step.outputs):
                    # Inference finals are published atomically, but no completion marker means
                    # we cannot silently certify them. Keep artifacts and require inspection.
                    raise ValueError(f'Unrecorded final output for {step.name}; inspect before recovery')
                log=root/f'logs/{step.name}.log'
                if log.exists(): log.rename(log.with_name(log.name+f'.previous_{time.time_ns()}'))
                atomic_json(root/'status.json',dict(status='running',stage=step.name))
                print('Starting',step.name,flush=True);start=time.monotonic()
                with log.open('w') as stream:
                    subprocess.run(step.command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True)
                if not all(Path(p).exists() for p in step.outputs): raise ValueError('Stage did not produce all outputs')
                state[step.name]=dict(signature=signature,inputs=input_hashes,
                    outputs={p:fingerprint(p) for p in step.outputs},seconds=time.monotonic()-start)
                atomic_json(state_path,state);print('Completed',step.name,flush=True)
            atomic_json(root/'status.json',dict(status='complete' if 'report' in state else 'selected_stages_complete'))
        except Exception as exc:
            atomic_json(root/'status.json',dict(status='failed',error=str(exc)));raise


def resolve_stages(requested,steps):
    """Expand public DP3O stage groups using only runs in the experiment profile."""
    if requested is None: return None
    names=[step.name for step in steps]
    aliases={name:[stage for stage in names if stage.startswith(name+'_')]
             for name in ['dp3o','audit_dp3o','generate_dp3o','score_dp3o']}
    selected=set()
    for name in requested:
        if name in aliases and aliases[name]: selected.update(aliases[name])
        elif name in names: selected.add(name)
        else: raise ValueError(f'Unknown stage: {name}')
    return [name for name in names if name in selected]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=REPO/'configs/experiment.yaml')
    p.add_argument('--run-name',required=True)
    p.add_argument('--smoke',action='store_true',help='Full model sizes; 16/8/8 rows and 2 training steps, not a result reproduction')
    p.add_argument('--cpu-offload',action='store_true',help='Smoke only: torch AdamW optimizer states on CPU for shared GPUs')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--stages',nargs='+',help='Exact stages or groups: dp3o, audit_dp3o, generate_dp3o, score_dp3o; prerequisites must exist')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_name): p.error('Invalid run name')
    a.config=a.config.resolve();root=external_path(Path(os.environ['DP3O_STORAGE'])/'runs'/a.run_name,'run directory')
    steps=plan(root,a.config,a.smoke,a.cpu_offload)
    try: a.stages=resolve_stages(a.stages,steps)
    except ValueError as exc: p.error(str(exc))
    if a.dry_run:
        print(json.dumps(dict(root=str(root),steps=[asdict(s) for s in steps if a.stages is None or s.name in a.stages]),indent=2));return
    execute(root,steps,a)


if __name__=='__main__': main()
