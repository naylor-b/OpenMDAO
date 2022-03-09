"""
This allows you to run a specific test function by itself from an importable module.

The function can be either a method of a unittest.TestCase subclass, or just a function defined
at module level.  This is useful when running an individual test under mpirun for debugging
purposes.

To specify the test to run, use the following forms:

<modpath>:<testcase name>.<funcname>   OR   <modpath>:<funcname>

where <modpath> is either the dotted module name or the full filesystem path of the python file.

for example:

    mpirun -n 4 run_test mypackage.mysubpackage.mymod:MyTestCase.test_foo

    OR

    mpirun -n 4 run_test /foo/bar/mypackage/mypackage/mysubpackage/mymod.py:MyTestCase.test_foo
"""

import sys
import os
import importlib
import time
import argparse
from openmdao.utils.general_utils import do_nothing_context
from openmdao.utils.file_utils import get_module_path
from openmdao.devtools.debug import profiling
from openmdao.utils.mpi import MPI


def run_test():
    """
    Run individual test(s).
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loops', action='store', dest='loops',
                        default='1', type=int,
                        help='Determines how many times the test will be run.')
    parser.add_argument('-p', '--profile', action='store_true', dest='profile',
                        help='If True, run profiler on the test.')
    parser.add_argument('testspec', metavar='testspec', nargs=1,
                        help='Testspec indicating which test function to run. Should be of the '
                        'form: mod_path:testcase.test_func or mod_path:test_func.')

    options = parser.parse_args()

    testspec = options.testspec[0]
    parts = testspec.split(':')

    sys.path.insert(0, os.path.dirname(parts[0]))

    modpath, funcpath = parts
    if modpath.endswith('.py'):
        modpath = get_module_path(modpath)

    mod = importlib.import_module(modpath)

    parts = funcpath.split('.', 1)
    funcs = []
    if len(parts) == 2:
        tcase_name, method_name = parts
        testcase = getattr(mod, tcase_name)(methodName=method_name)
        setup = getattr(testcase, 'setUp', None)
        if setup is not None:
            funcs.append(setup)
        funcs.append(getattr(testcase, method_name))
        teardown = getattr(testcase, 'tearDown', None)
        if teardown:
            funcs.append(teardown)
    else:
        funcname = parts[0]
        funcs.append(getattr(mod, funcname))

    rank = MPI.COMM_WORLD.rank if MPI else 0

    start = time.time()
    with profiling(f"prof{rank}.out") if options.profile else do_nothing_context():
        for i in range(options.loops):
            for f in funcs:
                f()
    end = time.time()

    per_iter = (end - start)/options.loops
    print(f"Elapsed time: {end - start} over {options.loops} iterations  ({per_iter}/iteration).")



if __name__ == '__main__':
    run_test()
