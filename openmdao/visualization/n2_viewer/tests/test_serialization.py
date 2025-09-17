"""
This test assures that the use of nonstandard datatypes anywhere that they are allowed in a model
does not break the recording of the JSON data structures needed for the model viewer.
"""

import unittest

import numpy as np
from pydantic import Field

import openmdao.api as om
from openmdao.core.driver import Driver, _DriverOptions, _DriverModel
from openmdao.core.explicitcomponent import _ExplicitComponentOptions, _ExplicitComponentModel
from openmdao.core.implicitcomponent import _ImplicitComponentOptions, _ImplicitComponentModel
from openmdao.utils.validation import DataModelManager as dmm
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.solvers.nonlinear.nonlinear_runonce import _NonlinearRunOnceOptions
from openmdao.solvers.linear.linear_block_gs import _NonIterLinearBlockGSOptions


class BadOpt(object):
    def junk(self):
        pass


class NonSerComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x', np.zeros((6, )),
                       tags=['a', 'c'])
        self.add_output('y', np.zeros((6, )),
                       tags=['a', 'b'])

        self.add_discrete_input('dx', [{('Discrete_i', BadOpt): (2, {((1, ), (2, )): 'stuff'})}])
        self.add_discrete_output('dy', [{('Discrete_o', BadOpt): (2, {((1, ), (2, )): 'stuff'})}])
        self.add_discrete_input('dcomplex', 3 + 5j)

class NonSerCompOptions(_ExplicitComponentOptions):
    good: str = Field(default='good_string', desc='Good option')
    bad: list = Field(default=[{(1, BadOpt): (2, 3)}], desc='Bad option')
    bad2: dict = Field(default={((1, ), (2, )): 'stuff'}, desc='Bad option 2')
    nonrec: float = Field(default=3.0, exclude=True, desc='Non-recordable option')
    cx: complex = Field(default=3 + 7j, desc='Complex option')

@dmm.register(NonSerComp)
class NonSerCompModel(_ExplicitComponentModel):
    options: NonSerCompOptions = Field(default_factory=NonSerCompOptions)


class NonSerIComp(om.ImplicitComponent):

    def setup(self):
        self.add_input('xx', np.zeros((6, )))
        self.add_output('yy', np.zeros((6, )))

        self.add_discrete_input('problem', None)

class NonSerICompOptions(_ImplicitComponentOptions):
    good: str = Field(default='good_string', desc='Good option')
    bad: list = Field(default=[{(1, BadOpt): (2, 3)}], desc='Bad option')
    bad2: dict = Field(default={((1, ), (2, )): 'stuff'}, desc='Bad option 2')
    nonrec: float = Field(default=3.0, exclude=True, desc='Non-recordable option')
    problem: object = Field(default=None, desc='Problem option')

@dmm.register(NonSerIComp)
class NonSerICompModel(_ImplicitComponentModel):
    options: NonSerICompOptions = Field(default_factory=NonSerICompOptions)


class _NonSerNLOptions(_NonlinearRunOnceOptions):
    bad: list[tuple] = Field([{(1, BadOpt): (2, 3)}])
    bad2: dict[tuple, str] = Field({((1, ), (2, )): 'stuff'})
    nonrec: float = Field(3.0, exclude=True)

class NonSerNL(om.NonlinearRunOnce):
    options: _NonSerNLOptions = Field(default_factory=_NonSerNLOptions)



class _NonSerLNOptions(_NonIterLinearBlockGSOptions):
    bad: list[tuple] = Field([{(1, BadOpt): (2, 3)}])
    bad2: dict[tuple, str] = Field({((1, ), (2, )): 'stuff'})
    nonrec: float = Field(3.0, exclude=True)


class NonSerLN(om.LinearRunOnce):
    options: _NonSerLNOptions = Field(default_factory=_NonSerLNOptions)



class NonSerDriver(Driver):
    pass


class NonSerDriverOptions(_DriverOptions):
    bad: list[tuple] = Field([{(1, BadOpt): (2, 3)}])
    bad2: dict[tuple, str] = Field({((1, ), (2, )): 'stuff'})
    nonrec: float = Field(3.0, exclude=True)


@dmm.register(NonSerDriver)
class NonSerDriverModel(_DriverModel):
    options: NonSerDriverOptions = Field(default_factory=NonSerDriverOptions)


@use_tempdirs
class TestSerialization(unittest.TestCase):

    def test_exhaustive_model(self):
        em_prob = om.Problem()
        em_model = em_prob.model
        em_model.add_subsystem('ns', NonSerComp())
        em_model.add_subsystem('nsi', NonSerIComp())
        em_prob.setup()
        em_prob.final_setup()

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('ns', NonSerComp())
        nsi = model.add_subsystem('nsi', NonSerIComp())

        nsi.options['problem'] = em_prob

        model.nonlinear_solver = NonSerNL()
        model.linear_solver = NonSerLN()
        nsi.nonlinear_solver = NonSerNL()
        nsi.linear_solver = NonSerLN()

        model.add_design_var('ns.x', indices=om.slicer[2:])
        model.add_design_var('ns.dx')
        model.add_constraint('ns.y', indices=om.slicer[2:])
        model.add_objective('nsi.yy', index=om.slicer[-1])

        prob.driver = NonSerDriver()

        prob.add_recorder(om.SqliteRecorder("cases1.sql"))
        prob.driver.add_recorder(om.SqliteRecorder("cases2.sql"))
        prob.model.add_recorder(om.SqliteRecorder("cases3.sql"))
        prob.model.nonlinear_solver.add_recorder(om.SqliteRecorder("cases4.sql"))

        prob.setup()
        prob.set_val('nsi.problem', em_prob)

        prob.run_model()

        cr = om.CaseReader(prob.get_outputs_dir() / "cases1.sql")

        dval = cr.problem_metadata['tree']['children'][1]['options']['bad']
        key, val = [(k, v) for k, v in dval[0].items()][0]
        self.assertTrue("(1, <class" in key)
        self.assertEqual(val, [2, 3])

        self.assertEqual(cr.problem_metadata['tree']['children'][1]['options']['bad2'],
                         {'((1,), (2,))': 'stuff'})
        self.assertEqual(cr.problem_metadata['tree']['children'][1]['options']['nonrec'],
                         'Not Recordable')
        self.assertEqual(cr.problem_metadata['tree']['children'][1]['options']['cx'],
                         '(3+7j)')

        dval = cr.problem_metadata['tree']['children'][1]['children'][4]['val']
        key, val = [(k, v) for k, v in dval[0].items()][0]
        self.assertTrue("Discrete_o', <class" in key)
        self.assertEqual(val, [2, {'((1,), (2,))': 'stuff'}])

        dval = cr.problem_metadata['tree']['children'][1]['children'][1]['val']
        key, val = [(k, v) for k, v in dval[0].items()][0]
        self.assertTrue("Discrete_i', <class" in key)
        self.assertEqual(val, [2, {'((1,), (2,))': 'stuff'}])

        self.assertEqual(cr.problem_metadata['tree']['children'][1]['children'][2]['val'],
                         '(3+5j)')


if __name__ == "__main__":
    unittest.main()
