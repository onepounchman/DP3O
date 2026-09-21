"""Read literal .env settings before importing ML libraries; keep artifacts external."""
import argparse
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]


def read_env(path):
    result = {}
    if not path.is_file():
        raise ValueError(f'Missing environment file: {path}. Create it and set DP3O_STORAGE to an absolute external directory.')
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.removeprefix('export ').partition('=')
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r'[A-Z][A-Z0-9_]*', key):
            raise ValueError(f'Invalid .env assignment on line {number}')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        result[key] = value
    return result


def external_path(value, label):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f'{label} must be absolute')
    path = path.resolve()
    if path == REPO or REPO in path.parents or path == Path('/'):
        raise ValueError(f'{label} must be outside the codebase, not a filesystem root')
    return path


def environment():
    env = dict(os.environ)
    config = Path(env.get('DP3O_ENV_FILE', REPO/'.env')).expanduser().resolve()
    # Exported variables take precedence; never log values of credentials.
    for key, value in read_env(config).items():
        if value and not env.get(key):
            env[key] = value
    storage = external_path(env.get('DP3O_STORAGE', ''), 'DP3O_STORAGE')
    venv = external_path(env.get('DP3O_VENV') or storage/'venv', 'DP3O_VENV')
    hf = external_path(env.get('HF_HOME') or storage/'cache/huggingface', 'HF_HOME')
    paths = dict(DP3O_STORAGE=storage, DP3O_VENV=venv, HF_HOME=hf,
                 HF_HUB_CACHE=hf/'hub', HF_DATASETS_CACHE=hf/'datasets',
                 XDG_CACHE_HOME=storage/'cache', TORCH_HOME=storage/'cache/torch',
                 PIP_CACHE_DIR=storage/'cache/pip', UV_CACHE_DIR=storage/'cache/uv',
                 TMPDIR=storage/'tmp', TRITON_CACHE_DIR=storage/'cache/triton',
                 TORCH_EXTENSIONS_DIR=storage/'cache/torch_extensions',
                 WANDB_DIR=storage/'wandb', PYTHONPYCACHEPREFIX=storage/'cache/pycache')
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['DP3O_ENV_FILE'] = str(config)
    env['PYTHONPATH'] = str(REPO/'src') + os.pathsep + str(REPO/'scripts')
    env['PYTHONNOUSERSITE'] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env.setdefault('WANDB_MODE', 'disabled')
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('DP3O_NUM_GPUS', '4')
    env['PATH'] = str(venv/'bin') + os.pathsep + env['PATH']
    return env, venv


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run','python'])
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    return p.parse_args(argv)


def main():
    args = parse_args()
    rest = args.arguments
    env, venv = environment()
    python = venv/'bin/python'
    if not python.is_file():
        raise ValueError('Run bash setup/install.sh first')
    commands = {'run':[str(REPO/'scripts/run_pipeline.py')], 'python':[]}
    os.chdir(REPO)
    os.execve(str(python), [str(python), *commands[args.action], *rest], env)


if __name__ == '__main__':
    main()
