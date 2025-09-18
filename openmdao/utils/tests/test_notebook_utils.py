""" Unit tests for the notebook_utils."""

import unittest
from pydantic import Field, BaseModel

try:
    import IPython
except ImportError:
    IPython = False

import openmdao.api as om
from openmdao.utils.assert_utils import assert_warning
from openmdao.utils.validation import _OptionsBaseModel

class _StateOptions(_OptionsBaseModel):
    name: str = Field(default='foo', desc='name of ODE state variable')


class _StateModel(BaseModel):
    options: _StateOptions = Field(default_factory=_StateOptions)


@unittest.skipUnless(IPython, "IPython is required")
class TestNotebookUtils(unittest.TestCase):

    @unittest.skipIf(not IPython, reason='Test requires IPython')
    def test_show_obj_options(self):
        from openmdao.utils import notebook_utils
        notebook_utils.ipy = True
        try:
            om.show_options_table("openmdao.utils.tests.test_notebook_utils._StateModel")
        except Exception as e:
            self.fail('show_options_table raised the following exception:\n' + str(e))

    def test_show_options_w_attr(self):
        from openmdao.utils import notebook_utils
        notebook_utils.ipy = True

        options = om.show_options_table("openmdao.components.balance_comp.BalanceComp")

        self.assertEqual(options, None)

    def test_show_options_table_warning(self):
        from openmdao.utils import notebook_utils
        notebook_utils.ipy = False

        msg = ("IPython is not installed. Run `pip install openmdao[notebooks]` or `pip install "
               "openmdao[docs]` to upgrade.")

        with assert_warning(UserWarning, msg):
            om.show_options_table("openmdao.components.balance_comp.BalanceComp")