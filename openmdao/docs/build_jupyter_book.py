#!/usr/bin/env python
import argparse
import os
import shutil
import subprocess
import pathlib

from copy_build_artifacts import copy_build_artifacts
from build_source_docs import build_src_docs

_this_file = pathlib.Path(__file__).resolve()
DOC_ROOT = _this_file.parent
BOOK_DIR = pathlib.Path(DOC_ROOT, 'openmdao_book')


def build_book(book_dir, clean, keep_going, config):
    """
    Clean (if requested), build, and copy over necessary files for the JupyterBook to be created.
    Parameters
    ----------
    book_dir
    clean
    """
    save_cwd = os.getcwd()
    os.chdir(DOC_ROOT)

    if clean:
        print("Cleaning out old _srcdocs, _build, and output artifacts...")
        try:
            shutil.rmtree(os.path.join(book_dir, '_srcdocs'))
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree(os.path.join(book_dir, '_build'))
        except FileNotFoundError:
            pass
        subprocess.run(['jupyter-book', 'clean', book_dir])  # nosec: trusted input

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(book_dir)))
    build_src_docs(book_dir, repo_dir, clean=clean)

    cmd = ['jupyter-book', 'build', '-W']
    if keep_going:
        cmd.append('--keep-going')
    if config:
        cmd.append(f'--config={config}')
    cmd.append(book_dir)

    try:
       subprocess.run(cmd)  # nosec: trusted input
    finally:
        copy_build_artifacts(book_dir)  # copy what we managed to build

    os.chdir(save_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a JupyterBook and automatically copy over'
                                                 'necessary build artifacts')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='Clean the old book out before building (default is False).')
    parser.add_argument('-k', '--keep_going', action='store_true',
                        help='Keep going after errors.')
    parser.add_argument('-b', '--book', action='store', default='openmdao_book',
                        help="The name of the book to be built (default is 'openmdao_book').")
    parser.add_argument('--config', action='store',
                        help="Specify and alternate config file.")
    args = parser.parse_args()

    build_book(book_dir=args.book, clean=args.clean, keep_going=args.keep_going, config=args.config)


