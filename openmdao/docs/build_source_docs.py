import os
import shutil
import json

packages = [
    'approximation_schemes',
    'components',
    'core',
    'drivers',
    'error_checking',
    'jacobians',
    'matrices',
    'proc_allocators',
    'recorders',
    'solvers.linear',
    'solvers.linesearch',
    'solvers.nonlinear',
    'surrogate_models',
    'test_suite.components',
    'test_suite.scripts',
    'utils',
    'vectors',
    'visualization',
]


def header(filename, path):
    return ("""# %s

```{eval-rst}
    .. automodule:: %s
        :undoc-members:
        :special-members: __init__, __contains__, __iter__, __setitem__, __getitem__
        :show-inheritance:
        :inherited-members:
        :noindex:
```
""" % (filename, path)).splitlines(keepends=True)


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


def build_src_docs(top, src_dir, project_name='openmdao', clean=False):

    doc_dir = os.path.join(top, "_srcdocs")
    if clean and os.path.isdir(doc_dir):
        shutil.rmtree(doc_dir)

    if not os.path.isdir(doc_dir):
        os.mkdir(doc_dir)

    packages_dir = os.path.join(doc_dir, "packages")
    if not os.path.isdir(packages_dir):
        os.mkdir(packages_dir)

    index_filename = os.path.join(doc_dir, "index.ipynb")
    index_data = ["\n# Source Docs\n\n"]

    for package in packages:
        # a package is e.g. openmdao.core, that contains source files
        # a sub_package, is a src file, e.g. openmdao.core.component
        sub_packages = []
        package_filename = os.path.join(packages_dir, package + ".ipynb")
        package_name = project_name + "." + package

        # the sub_listing is going into each package dir and listing what's in it
        package_dir = os.path.join(src_dir, package.replace('.','/'))
        for sub_listing in sorted(os.listdir(package_dir)):
            # don't want to catalog files twice, nor use init files nor test dir
            if (os.path.isdir(sub_listing) and sub_listing != "tests") or \
                (sub_listing.endswith(".py") and not sub_listing.startswith('_')):
                # just want the name of e.g. dataxfer not dataxfer.py
                sub_packages.append(sub_listing.rsplit('.')[0])

        if len(sub_packages) > 0:
            # continue to write in the top-level index file.
            # only document non-empty packages -- to avoid errors
            # (e.g. at time of writing, doegenerators, drivers, are empty dirs)

            # specifically don't use os.path.join here.  Even windows wants the
            # stuff in the file to have fwd slashes.
            index_data.append(f"- [{package}](packages/{package}.md)\n\n")

            # make subpkg directory (e.g. _srcdocs/packages/core) for ref sheets
            package_dir = os.path.join(packages_dir, package)
            if not os.path.isdir(package_dir):
                os.mkdir(package_dir)

            # create/write a package index file: (e.g. "_srcdocs/packages/openmdao.core.ipynb")
            package_data = [f"# {package_name}\n\n"]

            SKIP_SUBPACKAGES = ['__pycache__']
            for sub_package in sub_packages:
                # this line writes subpackage name e.g. "core/component.py"
                # into the corresponding package index file (e.g. "openmdao.core.ipynb")
                if sub_package not in SKIP_SUBPACKAGES:
                    # specifically don't use os.path.join here.  Even windows wants the
                    # stuff in the file to have fwd slashes.
                    package_data.append(f"- [{sub_package}]({package}/{sub_package}.md)\n\n")

                    # creates and writes out one reference sheet (e.g. core/component.ipynb)
                    ref_sheet_filename = os.path.join(package_dir, sub_package + ".ipynb")

                    data = _header_cell()
                    data['cells'][0]['source'] = header(sub_package + ".py",
                                                        package_name + "." + sub_package)

                    # Create the json file
                    with open(ref_sheet_filename, 'w') as f:
                        json.dump(data, f, indent=4)

            # finish and close each package file
            data = _header_cell()
            data['cells'][0]['source'] = package_data
            with open(package_filename, 'w') as f:
                json.dump(data, f, indent=4)

    # finish and close top-level index file
    data = _header_cell()
    with open(index_filename, 'w') as f:
        data['cells'][0]['source'] = index_data
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    import sys
    build_src_docs("openmdao_book/", "..", clean='clean' in sys.argv)
