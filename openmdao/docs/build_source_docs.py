import os
import shutil
import json
from pathlib import Path
from hashlib import blake2b

from openmdao.utils.file_utils import list_package_pyfiles, get_module_path


def header(basename, modpath):
    return ("""# %s

```{eval-rst}
    .. automodule:: %s
        :undoc-members:
        :special-members: __init__, __contains__, __iter__, __setitem__, __getitem__
        :show-inheritance:
        :inherited-members:
        :noindex:
```
""" % (basename, modpath)).splitlines(keepends=True)


def _header_cell():
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    ""
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.1"
            },
            "orphan": True
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


skip_packages = {
    'docs',
    'devtools',
    'code_review',
    'tests',
    'test',
    'test_examples'
}


def _get_hash(s):
    h = blake2b()
    h.update(s)
    return h.hexdigest()


def _hash_changed(fpath, newhash):
    with open(fpath, 'rb') as f:
        contents = f.read()
    return newhash != _get_hash(contents)


def build_src_docs(doc_top, project_name='openmdao', clean=False):

    doc_dir = Path(doc_top, "_srcdocs")
    if clean and doc_dir.is_dir():
        shutil.rmtree(doc_dir)

    doc_dir.mkdir(parents=True, exist_ok=True)

    packages_dir = doc_dir.joinpath("packages")
    packages_dir.mkdir(parents=True, exist_ok=True)

    # entries are (index_file, index_lines, pyfiles)
    packages = {
        project_name: (doc_dir.joinpath("index.ipynb"), ["\n# Source Docs\n\n"], [])
    }
    IDX_LINES, PYFILES = 1, 2

    # these packages are hidden but can still recursed into
    hide_pkgs = {'', 'test_suite', 'test_suite.groups'}

    for pyfile in list_package_pyfiles(project_name, file_excludes={'__init__.py'},
                                       dir_excludes=skip_packages):
        modpath = get_module_path(pyfile)
        package = '.'.join(modpath.split('.')[1:-1])
        if package in hide_pkgs:
            continue
        if package not in packages:
            package_name = '.'.join((project_name, package))
            packages[package] = (packages_dir.joinpath(package + ".ipynb"),
                                 [f"# {package_name}\n\n"], [])
        packages[package][PYFILES].append(Path(pyfile))

    for package, (_, idxlines, pyfiles) in sorted(packages.items(), key=lambda x: x[0]):
        parts = package.split('.')
        parent = project_name if len(parts) == 1 else '.'.join(parts[:-1])

        if parent not in packages:
            parent = project_name

        if package != project_name:
            # specifically don't use a path join here.  Even windows wants the
            # stuff in the file to have fwd slashes.
            if parent == project_name:
                packages[parent][IDX_LINES].append(f"- [{package}](packages/{package}.ipynb)\n\n")
            else:
                packages[parent][IDX_LINES].append(f"- [{package}]({package}.ipynb)\n\n")

        for f in pyfiles:
            pname = f.stem
            if package == project_name:
                idxlines.append(f"- [{pname}](packages/{pname}.ipynb)\n\n")
            else:
                idxlines.append(f"- [{pname}]({package}/{pname}.ipynb)\n\n")

    top_level_old_nbs = set(os.path.join(packages_dir, f) for f in os.listdir(packages_dir) if f.endswith('.ipynb'))
    top_level_current = set()

    for package, (idxfile, idxlines, pyfiles) in packages.items():
        # make subpkg directory (e.g. _srcdocs/packages/core) for ref sheets
        if package == project_name:
            continue

        package_dir = packages_dir.joinpath(package)
        package_dir.mkdir(exist_ok=True)

        old_nbs = set(os.path.join(package_dir, f) for f in os.listdir(package_dir) if f.endswith('.ipynb'))
        new_nbs = set()
        
        max_pkg_mtime = 0
        for pyfile in pyfiles:
            py_mtime = pyfile.stat().st_mtime
            if py_mtime > max_pkg_mtime:
                max_pkg_mtime = py_mtime

            pybase = pyfile.name
            pname = pyfile.stem
            nbpath = package_dir.joinpath(pname + ".ipynb")
            new_nbs.add(str(nbpath))

            # only write the file if it doesn't exist or is older than the corresponding src file
            # we use mtime for the source file since we don't have its old hash, and we can't 
            # compare the hash of the new notebook file to the that of the old notebook file
            # because the code in there is dynamic sphinx code and could generate different docs 
            # based on changes in the original source file, so this is the best we can do.
            if not nbpath.exists() or nbpath.stat().st_mtime < py_mtime:
                data = _header_cell()
                data['cells'][0]['source'] = header(pybase,
                                                    '.'.join((project_name, package, pname)))
                print("writing notebook", str(nbpath))
                nbpath.write_text(json.dumps(data, indent=4))

        old_remaining = old_nbs - new_nbs
        for f in old_remaining:
            print("removing old noteboook", f)
            os.remove(f)

        # finish and close each package index file
        nbpath = Path(idxfile)
        if nbpath.parent == packages_dir:
            top_level_current.add(str(nbpath))
            
        data = _header_cell()
        data['cells'][0]['source'] = idxlines
        contents = json.dumps(data, indent=4)
        # for index files we compare hash of old to new
        if not nbpath.exists() or _hash_changed(nbpath, _get_hash(contents.encode('utf-8'))):
            print("writing index notebook", idxfile)
            nbpath.write_text(contents)

    old_remaining = top_level_old_nbs - top_level_current
    for f in old_remaining:
        print("removing old noteboook", f)
        os.remove(f)
        
    # do the top level index file last
    idxfile, idxlines, pyfiles = packages[project_name]
    nbpath = Path(idxfile)
    data = _header_cell()
    data['cells'][0]['source'] = idxlines
    contents = json.dumps(data, indent=4)
    # for index files we compare hash of old to new
    if not nbpath.exists() or _hash_changed(nbpath, _get_hash(contents.encode('utf-8'))):
        print("writing index notebook", idxfile)
        nbpath.write_text(contents)


if __name__ == '__main__':
    import sys
    build_src_docs("openmdao_book", clean='clean' in sys.argv)
