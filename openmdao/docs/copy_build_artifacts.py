#!/usr/bin/env python
import os
from pathlib import Path
import shutil

def copy_build_artifacts(book_dir='openmdao_book', verbose=False):
    """
    Copy build artifacts (html files, images, etc) to the output _build directory.

    Parameters
    ----------
    book_dir : str
        The directory containing the Jupyter-Book to be created.
    verbose : bool
        If True, print names of files being copied.
    """
    SUFFIXES_TO_COPY = ('.png', '.html')
    EXCLUDE_DIRS = {'_build', '.ipynb_checkpoints', '_srcdocs'}

    dest_parent = Path(book_dir, '_build', 'html')

    for dirpath, dirs, files in os.walk(book_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        target_path = dest_parent.joinpath(*Path(dirpath).parts[1:])

        for f in files:
            for suffix in SUFFIXES_TO_COPY:
                if f.endswith(suffix):
                    break
            else:
                continue  # didn't match

            src = Path(dirpath, f)
            dst = Path(target_path, f)
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if verbose:
                    print('copying to', str(dst))
                shutil.copyfile(src, dst)


if __name__ == '__main__':
    import sys
    copy_build_artifacts('openmdao_book', 'verbose' in sys.argv)


