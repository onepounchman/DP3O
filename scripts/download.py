"""Download pinned base models into external HF cache; no weights bundled."""
import argparse
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download
from hh_eval_utils import atomic_json
from bootstrap import external_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();external_path(a.output,'download manifest');cfg=yaml.safe_load(a.config.read_text())
    models={}
    for kind,entry in cfg['models'].items():
        print('Downloading/checking',kind,entry['name'],entry['revision'],flush=True)
        path=snapshot_download(entry['name'],revision=entry['revision'],
            allow_patterns=['*.json','*.safetensors','tokenizer.model','*.txt','*.tiktoken'])
        if not list(Path(path).glob('*.safetensors')):
            raise ValueError(f'Missing model weights: {kind}')
        models[kind]=dict(entry,path=path)
    atomic_json(a.output,models)


if __name__=='__main__': main()
