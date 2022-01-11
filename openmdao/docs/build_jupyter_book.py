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


def build_book(book_dir, package, clean, keep_going, verbose):
    """
    Clean (if requested), build, and copy over necessary files for the JupyterBook to be created.

    Parameters
    ----------
    book_dir : str
        Directory where book is to be created
    package : str
        Name of the package to be documented.
    clean : bool
        If True, remove any old generated files before building book.
    keep_going : bool
        If True, keep going even if there are warnings/errors.
    verbose : bool
        If True, print detailed information while running.
    """
    save_cwd = os.getcwd()
    os.chdir(DOC_ROOT)

    if clean:
        if verbose:
            print("Cleaning out old _srcdocs, _build, and output artifacts...")
        for dirname in ('_srcdocs', '_build'):
            try:
                shutil.rmtree(pathlib.Path(book_dir, dirname))
            except FileNotFoundError:
                pass

        subprocess.run(['jupyter-book', 'clean', book_dir])  # nosec: trusted input

    build_src_docs(book_dir, package=package, clean=clean, verbose=verbose)

    cmd = ['jupyter-book', 'build', '-W']
    if keep_going:
        cmd.append('--keep-going')
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
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed information.')
    parser.add_argument('-b', '--book', action='store', default='openmdao_book',
                        help="The directory where the book is to be built "
                             "(default is 'openmdao_book').")
    parser.add_argument('-p', '--package', action='store', default='openmdao',
                        help="The name of the package to document (default is 'openmdao').")
    args = parser.parse_args()

    build_book(book_dir=args.book, package=args.package, clean=args.clean,
               keep_going=args.keep_going, verbose=args.verbose)


