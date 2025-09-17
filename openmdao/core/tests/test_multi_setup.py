
import unittest
from pydantic import Field

import openmdao.api as om
from openmdao.core.group import _GroupOptions, _GroupModel
from openmdao.utils.validation import DataModelManager as dmm


class DiscreteOut1(om.ExplicitComponent):
    def setup(self):
        self.add_discrete_output('y', val=1)


class DiscreteIn1(om.ExplicitComponent):
    def setup(self):
        self.add_discrete_input('x', val=0)


class DiscreteIn2(om.ExplicitComponent):
    def setup(self):
        self.add_discrete_input('x', val=0)


class Group1Options(_GroupOptions):
    conn: int = Field(default=1, values=[1, 2], desc='Connection option')


class Group1(om.Group):

    def setup(self):
        self.add_subsystem('out1', DiscreteOut1())
        if self.options['conn'] == 1:
            self.add_subsystem('in1', DiscreteIn1())
            self.connect('out1.y', 'in1.x')
        else:
            self.add_subsystem('in2', DiscreteIn2())
            self.connect('out1.y', 'in2.x')


@dmm.register(Group1)
class Group1Model(_GroupModel):
    options: Group1Options = Field(default_factory=Group1Options)


class MultiSetupTestCase(unittest.TestCase):
    def test_multi_setup_discrete(self):
        prob = om.Problem()
        group = prob.model.add_subsystem('g1', Group1(conn=1))

        prob.setup()

        prob['g1.out1.y'] = 2

        prob.run_model()

        self.assertEqual(prob['g1.in1.x'], 2)

        group.options['conn'] = 2

        prob.setup()

        prob['g1.out1.y'] = 5

        prob.run_model()

        self.assertEqual(prob['g1.in2.x'], 5)
