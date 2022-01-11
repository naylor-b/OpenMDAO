#!/usr/bin/env python
import os
from pathlib import PurePath
import shutil

def copy_build_artifacts(book_dir='openmdao_book'):
    """
    Copy build artifacts (html files, images, etc) to the output _build directory.
    Parameters
    ----------
    book_dir : str
        The directory containing the Jupyter-Book to be created.
    """
    SUFFIXES_TO_COPY = ('.png', '.html')
    EXCLUDE_DIRS = {'_build', '.ipynb_checkpoints', '_srcdocs'}

    dest_parent = PurePath(book_dir, '_build', 'html')

    for dirpath, dirs, files in os.walk(book_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        target_path = dest_parent.joinpath(*PurePath(dirpath).parts[1:])

        for f in files:
            for suffix in SUFFIXES_TO_COPY:
                if f.endswith(suffix):
                    break
            else:
                continue  # didn't match

            src = PurePath(dirpath, f)
            dst = PurePath(target_path, f)
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)


if __name__ == '__main__':
    copy_build_artifacts('openmdao_book')


