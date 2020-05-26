import unittest
import time
from collections.abc import Iterable

import numpy as np

import openmdao.api as om
from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import evenly_distrib_idxs, take_nth
from openmdao.utils.assert_utils import assert_near_equal, assert_warning

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


def _test_func_name(func, num, param):
    args = []
    for p in param.args:
        if isinstance(p, str):
            p = {p.replace('.', '_')}
        elif not isinstance(p, Iterable):
            p = {p}
        for item in p:
            try:
                arg = item.__name__
            except:
                arg = str(item)
            args.append(arg)
    return func.__name__ + '_' + '_'.join(args)


class SerialTests(unittest.TestCase):
    @parameterized.expand([(3, 'par.C1.x', True),
                           (3, 'par.C2.x', True),
                           (3, 'indeps.x', False)],
                          name_func=_test_func_name)
    def test_fan_out(self, size, toset, auto):
        p = om.Problem()
        model = p.model
        if not auto:
            ivc = model.add_subsystem('indeps', om.IndepVarComp())
            ivc.add_output('x', np.ones(size))

        par = model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('C1', om.ExecComp('y = 3 * x', x=np.ones(size), y=np.ones(size)), promotes_inputs=['x'])
        par.add_subsystem('C2', om.ExecComp('y = 5 * x', x=np.ones(size), y=np.ones(size)), promotes_inputs=['x'])

        if not auto:
            model.connect('indeps.x', 'par.x')

        p.setup()

        inval = np.arange(size) + 1.0

        p[toset] = inval
        p.run_model()

        np.testing.assert_allclose(p.get_val('par.C1.y', get_remote=True), inval * 3.)
        np.testing.assert_allclose(p.get_val('par.C2.y', get_remote=True), inval * 5.)

        of = ['par.C1.y', 'par.C2.y']
        if auto:
            wrt = ['par.x']
        else:
            wrt = ['indeps.x']

        J = p.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        np.testing.assert_allclose(J['par.C1.y', wrt[0]], 3. * np.eye(size))
        np.testing.assert_allclose(J['par.C2.y', wrt[0]], 5. * np.eye(size))

    @parameterized.expand([(3, 'par.C1.x', True, True), (3, 'par.C1.x', True, False),
                           (3, 'par.C2.x', True, True), (3, 'par.C2.x', True, False),
                           (3, 'C3.x', True, True), (3, 'C3.x', True, False),
                           (3, 'indeps.x', False, True), (3, 'indeps.x', False, False)],
                          name_func=_test_func_name)
    def test_fan_out_with_dup(self, size, toset, auto, before):
        # this connects an auto_ivc to 3 variables.  2 are under a parallel group and 1 is
        # duplicated in all procs
        p = om.Problem()
        model = p.model

        if not auto:
            ivc = model.add_subsystem('indeps', om.IndepVarComp())
            ivc.add_output('x', np.ones(size))

        c3 = om.ExecComp('y = 4. * x', x=np.ones(size), y=np.ones(size))

        if before:
            model.add_subsystem('C3', c3, promotes_inputs=['x'])

        par = model.add_subsystem('par', om.ParallelGroup(), promotes_inputs=['x'])
        par.add_subsystem('C1', om.ExecComp('y = 3 * x', x=np.ones(size), y=np.ones(size)), promotes_inputs=['x'])
        par.add_subsystem('C2', om.ExecComp('y = 5 * x', x=np.ones(size), y=np.ones(size)), promotes_inputs=['x'])

        if not before:
            model.add_subsystem('C3', c3, promotes_inputs=['x'])

        if not auto:
            model.connect('indeps.x', 'x')

        p.setup()

        inval = np.arange(size) + 1.0

        p[toset] = inval
        p.run_model()

        np.testing.assert_allclose(p.get_val('par.C1.y', get_remote=True), inval * 3.)
        np.testing.assert_allclose(p.get_val('par.C2.y', get_remote=True), inval * 5.)
        np.testing.assert_allclose(p.get_val('C3.y', get_remote=True), inval * 4.)

        of = ['par.C1.y', 'par.C2.y', 'C3.y']

        if auto:
            wrt = ['x']
        else:
            wrt = ['indeps.x']

        J = p.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        np.testing.assert_allclose(J['par.C1.y', wrt[0]], 3. * np.eye(size))
        np.testing.assert_allclose(J['par.C2.y', wrt[0]], 5. * np.eye(size))
        np.testing.assert_allclose(J['C3.y', wrt[0]], 4. * np.eye(size))

        # try with absolute names
        if not auto:
            J = p.compute_totals(of=of, wrt=['indeps.x'], return_format='flat_dict')
            np.testing.assert_allclose(J['par.C1.y', 'indeps.x'], 3. * np.eye(size))
            np.testing.assert_allclose(J['par.C2.y', 'indeps.x'], 5. * np.eye(size))
            np.testing.assert_allclose(J['C3.y', 'indeps.x'], 4. * np.eye(size))

    @parameterized.expand([(3, 'par.C1.x', 'par.C2.x', True),
                           (3, 'par.C1.x', 'par.C2.x', True),
                           (3, 'indeps1.x', 'indeps2.x', False)],
                          name_func=_test_func_name)
    def test_fan_in(self, size, toset1, toset2, auto):
        p = om.Problem()
        model = p.model
        if not auto:
            ivc = model.add_subsystem('indeps1', om.IndepVarComp())
            ivc.add_output('x', np.ones(size))
            ivc = model.add_subsystem('indeps2', om.IndepVarComp())
            ivc.add_output('x', np.ones(size))

        par = model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('C1', om.ExecComp('y = 3 * x', x=np.ones(size), y=np.ones(size)))
        par.add_subsystem('C2', om.ExecComp('y = 5 * x', x=np.ones(size), y=np.ones(size)))

        model.add_subsystem('sum', om.ExecComp('z = x + y', x=np.ones(size), y=np.ones(size), z=np.ones(size)))

        if not auto:
            model.connect('indeps1.x', 'par.C1.x')
            model.connect('indeps2.x', 'par.C2.x')

        model.connect('par.C1.y', 'sum.x')
        model.connect('par.C2.y', 'sum.y')

        p.setup()

        inval1 = np.arange(size) + 1.0
        inval2 = (np.arange(size) + 1.0)[::-1]

        p[toset1] = inval1
        p[toset2] = inval2

        p.run_model()
        np.testing.assert_allclose(p['sum.z'], inval1 * 3. + inval2 * 5.)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MPITests(SerialTests):

    N_PROCS = 2

