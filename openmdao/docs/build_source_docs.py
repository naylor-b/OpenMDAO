import os
import shutil
import json
from openmdao.utils.file_utils import list_package_pyfiles, get_module_path
import pathlib


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


def build_src_docs(doc_top, src_dir, project_name='openmdao', clean=False):

    doc_dir = os.path.join(doc_top, "_srcdocs")
    if clean and os.path.isdir(doc_dir):
        shutil.rmtree(doc_dir)

    if not os.path.isdir(doc_dir):
        os.mkdir(doc_dir)

    packages_dir = os.path.join(doc_dir, "packages")
    if not os.path.isdir(packages_dir):
        os.mkdir(packages_dir)

    # entries are (index_file, index_lines, pyfiles)
    packages = {
        project_name: (os.path.join(doc_dir, "index.ipynb"), ["\n# Source Docs\n\n"], [])
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
            package_name = project_name + "." + package
            packages[package] = (os.path.join(packages_dir, package + ".ipynb"), [f"# {package_name}\n\n"], [])
        packages[package][PYFILES].append(pyfile)

    for package, (_, idxlines, pyfiles) in sorted(packages.items(), key=lambda x: x[0]):
        parts = package.split('.')
        parent = project_name if len(parts) == 1 else '.'.join(parts[:-1])

        if parent not in packages:
            parent = project_name

        if package != project_name:
            # specifically don't use os.path.join here.  Even windows wants the
            # stuff in the file to have fwd slashes.
            if parent == project_name:
                packages[parent][IDX_LINES].append(f"- [{package}](packages/{package}.ipynb)\n\n")
            else:
                packages[parent][IDX_LINES].append(f"- [{package}]({package}.ipynb)\n\n")

        for f in pyfiles:
            pname = os.path.splitext(os.path.basename(f))[0]
            if package == project_name:
                idxlines.append(f"- [{pname}](packages/{pname}.ipynb)\n\n")
            else:
                idxlines.append(f"- [{pname}]({package}/{pname}.ipynb)\n\n")

    for package, (idxfile, idxlines, pyfiles) in packages.items():
        # make subpkg directory (e.g. _srcdocs/packages/core) for ref sheets
        package_dir = os.path.join(packages_dir, package)
        if package_dir != project_name and not os.path.isdir(package_dir):
            os.mkdir(package_dir)

        max_pkg_mtime = 0
        for pyfile in pyfiles:
            fpath = pathlib.Path(pyfile)
            py_mtime = fpath.stat().st_mtime
            if py_mtime > max_pkg_mtime:
                max_pkg_mtime = py_mtime

            pybase = os.path.basename(pyfile)
            pname = os.path.splitext(pybase)[0]
            notebook = os.path.join(package_dir, pname + ".ipynb")
            nbpath = pathlib.Path(notebook)

            # only write the file if it doesn't exist or is older than the corresponding src file
            if not nbpath.exists() or nbpath.stat().st_mtime < py_mtime:
                data = _header_cell()
                data['cells'][0]['source'] = header(pybase,
                                                    '.'.join((project_name, package, pname)))
                print("writing", notebook)
                with open(notebook, 'w') as f:
                    json.dump(data, f, indent=4)

        # finish and close each package file
        nbpath = pathlib.Path(idxfile)
        if not nbpath.exists() or nbpath.stat().st_mtime < max_pkg_mtime:
            data = _header_cell()
            data['cells'][0]['source'] = idxlines
            print("writing", idxfile)
            with open(idxfile, 'w') as f:
                json.dump(data, f, indent=4)


if __name__ == '__main__':
    import sys
    build_src_docs("openmdao_book/", "..", clean='clean' in sys.argv)
