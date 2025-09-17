import unittest
from pydantic import Field

import openmdao.api as om
from openmdao.core.explicitcomponent import _ExplicitComponentOptions, _ExplicitComponentModel
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.validation import DataModelManager as dmm


class BadOptionComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x')
        self.add_output('y')

class BadOptionCompOptions(_ExplicitComponentOptions):
    bad: object = Field(default=object(), exclude=True, desc='Bad option')

@dmm.register(BadOptionComp)
class BadOptionCompModel(_ExplicitComponentModel):
    options: BadOptionCompOptions = Field(default_factory=BadOptionCompOptions)


@use_tempdirs
class SerializeTestCase(unittest.TestCase):
    def test_serialize_n2(self):
        p = om.Problem()

        p.model.add_subsystem('foo', BadOptionComp(bad=object()))

        p.setup()
        p.final_setup()

        om.n2(p, show_browser=False)

    def test_recordable_only(self):
        p = om.Problem()

        comp = p.model.add_subsystem('foo', BadOptionComp(bad=object()))

        p.setup()
        p.final_setup()

        # no errors
        opts_dict = dict(comp.options.items(recordable_only=True))

        # bad opt excluded
        self.assertTrue('bad' not in opts_dict)


if __name__ == '__main__':
    unittest.main()
