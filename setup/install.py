"""Install pinned dependencies and the local package into external storage."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from bootstrap import REPO, environment


def install(env, venv):
    if sys.version_info[:2] not in [(3, 10), (3, 11)]:
        raise ValueError('Use Python 3.10 or 3.11 (validated with 3.10)')
    python = venv/'bin/python'
    uv = shutil.which('uv')
    if not python.exists():
        command = [uv, 'venv', '--seed', '--python', sys.executable, str(venv)] if uv else [sys.executable, '-m', 'venv', str(venv)]
        subprocess.run(command, env=env, check=True)
    if uv:
        pip = [uv, 'pip']
        target = ['--python', str(python)]
    else:
        pip = [str(python), '-m', 'pip']
        target = []
    subprocess.run(pip+['install']+target+['-r', str(REPO/'requirements/base.txt')], env=env, check=True)
    # Build externally so setuptools never leaves build/egg-info in the source tree.
    with tempfile.TemporaryDirectory(prefix='dp3o_build_', dir=env['TMPDIR']) as temporary:
        staged=Path(temporary)/'package'
        shutil.copytree(REPO/'src',staged/'src',ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
        shutil.copy2(REPO/'pyproject.toml',staged/'pyproject.toml')
        subprocess.run(pip+['install']+target+['--no-deps', str(staged)], env=env, check=True)
    subprocess.run(pip+['check']+target, env=env, check=True)
    print('Environment installed.', flush=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    env, venv = environment()
    install(env, venv)


if __name__ == '__main__':
    main()
