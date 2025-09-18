import warnings
import unittest
from typing import Union
from pydantic import Field, ConfigDict, field_validator

from openmdao.utils.assert_utils import assert_warning, assert_no_warning
from openmdao.utils.om_warnings import OMDeprecationWarning

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.validation import _OptionsBaseModel


def check_even(name, value):
    if value % 2 != 0:
        raise ValueError("Option '%s' with value %s is not an even number." % (name, value))


class TestOptions(unittest.TestCase):

    def test_reprs(self):
        class MyComp(ExplicitComponent):
            pass

        my_comp = MyComp()

        class MyOptions(_OptionsBaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            test: str = Field(desc='Test integer value')
            flag: bool = Field(default=False)
            comp: ExplicitComponent = Field(default=my_comp)
            long_desc: str = Field(desc='This description is long and verbose, so it takes up multiple lines in the options table.')

        self.assertEqual(MyOptions(long_desc='', test='').__str__(width=89).strip(), """
=========  ============  ================================================================
Option     Default       Description
=========  ============  ================================================================
comp       MyComp
flag       False
long_desc  **Required**  This description is long and verbose, so it takes up multiple
                         lines in the options table.
test       **Required**  Test integer value
=========  ============  ================================================================
""".strip())

        # if the table can't be represented in specified width, then we get the full width version
        self.assertEqual(MyOptions(long_desc='', test='').__str__(width=30).strip(), """
=========  ============  =========================================================================================
Option     Default       Description
=========  ============  =========================================================================================
comp       MyComp
flag       False
long_desc  **Required**  This description is long and verbose, so it takes up multiple lines in the options table.
test       **Required**  Test integer value
=========  ============  =========================================================================================
""".strip())

    def test_to_table(self):
        class MyComp(ExplicitComponent):
            pass

        my_comp = MyComp()

        class MyOptions(_OptionsBaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            test: str = Field(desc='Test integer value')
            flag: bool = Field(default=False)
            comp: ExplicitComponent = Field(default=my_comp)
            long_desc: str = Field(desc='This description is long and verbose, so it takes up multiple lines in the options table.')

        expected = \
"""
| Option    | Default      | Description                                                                               |
| :-------- | :----------- | :---------------------------------------------------------------------------------------- |
| comp      | MyComp       |                                                                                           |
| flag      | False        |                                                                                           |
| long_desc | **Required** | This description is long and verbose, so it takes up multiple lines in the options table. |
| test      | **Required** | Test integer value                                                                        |
"""
        self.assertEqual(MyOptions(long_desc='', test='').to_table(fmt='github', display=False).strip(), expected.strip())

    def test_deprecation_col(self):
        class MyComp(ExplicitComponent):
            pass

        my_comp = MyComp()

        class MyOptions(_OptionsBaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            test: str = Field(desc='Test integer value')
            flag: bool = Field(default=False)
            comp: ExplicitComponent = Field(default=my_comp)
            long_desc: str = Field(desc='This description is long and verbose, so it takes up multiple lines in the options table.',
                                   deprecated='This option is deprecated')

        expected = \
"""
| Option    | Default      | Description                                                                               | Deprecation               |
| :-------- | :----------- | :---------------------------------------------------------------------------------------- | :------------------------ |
| comp      | MyComp       |                                                                                           | N/A                       |
| flag      | False        |                                                                                           | N/A                       |
| long_desc | **Required** | This description is long and verbose, so it takes up multiple lines in the options table. | This option is deprecated |
| test      | **Required** | Test integer value                                                                        | N/A                       |
"""

        self.assertEqual(MyOptions(long_desc='', test='').to_table(fmt='github', display=False).strip(), expected.strip())

        my_comp = MyComp()

        class MyOptions(_OptionsBaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            test: str = Field(desc='Test integer value')
            flag: bool = Field(default=False)
            comp: ExplicitComponent = Field(default=my_comp)
            long_desc: str = Field(desc='This description is long and verbose, so it takes up multiple lines in the options table.')

        expected = \
"""
| Option    | Default      | Description                                                                               |
| :-------- | :----------- | :---------------------------------------------------------------------------------------- |
| comp      | MyComp       |                                                                                           |
| flag      | False        |                                                                                           |
| long_desc | **Required** | This description is long and verbose, so it takes up multiple lines in the options table. |
| test      | **Required** | Test integer value                                                                        |
"""

        self.assertEqual(MyOptions(long_desc='', test='').to_table(fmt='github', display=False).strip(), expected.strip())

    def test_read_only(self):

        class MyOptions(_OptionsBaseModel):
            permanent: float = Field(default=3.0, frozen=True)

        opt = MyOptions()

        with self.assertRaises(Exception) as context:
            opt['permanent'] = 4.0

        msg = "permanent\n  Field is frozen [type=frozen_field, input_value=4.0, input_type=float]"
        self.assertTrue(msg in str(context.exception))

    def test_context_manager(self):
        class MyOptions(_OptionsBaseModel):
            foo: str = Field(default=None, desc='Test integer value')
            bar: Union[float, int] = Field(default=False)

        options = MyOptions()
        options['foo'] = 'b'
        options['bar'] = 3.14

        self.assertEqual(options['foo'], 'b')
        self.assertAlmostEqual(options['bar'], 3.14)

        with options.temporary(foo='c', bar=5):
            self.assertEqual(options['foo'], 'c')
            self.assertEqual(options['bar'], 5)
            with options.temporary(foo='a'):
                self.assertEqual(options['foo'], 'a')
            self.assertEqual(options['foo'], 'c')
            self.assertEqual(options['bar'], 5)

        self.assertEqual(options['foo'], 'b')
        self.assertAlmostEqual(options['bar'], 3.14)

    def test_call(self):
        class MyOptions(_OptionsBaseModel):
            foo: str = Field(default=None, desc='Test integer value')
            bar: Union[float, int] = Field(default=False)

        options = MyOptions()
        options['foo'] = 'b'
        options['bar'] = 3.14

        self.assertEqual(options['foo'], 'b')
        self.assertAlmostEqual(options['bar'], 3.14)
        options.set(foo='c', bar=5)
        self.assertEqual(options['foo'], 'c')
        self.assertEqual(options['bar'], 5)

    def test_type_checking(self):
        class MyOptions(_OptionsBaseModel):
            test: int = Field(default=0, desc='Test integer value')

        options = MyOptions()
        options['test'] = 1
        self.assertEqual(options['test'], 1)

        with self.assertRaises(Exception) as context:
            options['test'] = ''

        msg = "test\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='', input_type=str]"
        self.assertTrue(msg in str(context.exception))

        # multiple types are allowed
        class MyOptions(_OptionsBaseModel):
            test_multi: Union[int, float] = Field(default=0, desc='Test value')

        options = MyOptions()

        options['test_multi'] = 1
        self.assertEqual(options['test_multi'], 1)
        self.assertEqual(type(options['test_multi']), int)

        options['test_multi'] = 1.0
        self.assertEqual(options['test_multi'], 1.0)
        self.assertEqual(type(options['test_multi']), float)

        with self.assertRaises(Exception) as context:
            options['test_multi'] = ''

        msg = "test_multi.float\n  Input should be a valid number, unable to parse string as a number [type=float_parsing, input_value='', input_type=str]"
        self.assertTrue(msg in str(context.exception))

    def test_allow_none(self):
        class MyOptions(_OptionsBaseModel):
            test: Union[int, None] = Field(default=1, desc='Test value')

        options = MyOptions()
        options['test'] = None
        self.assertEqual(options['test'], None)

    def test_isvalid(self):
        class MyOptions(_OptionsBaseModel):
            even_test: int = Field(default=1, check_valid=check_even)

            @field_validator('even_test')
            @classmethod
            def _validate_even(cls, v):
                if not v % 2 == 0:
                    raise ValueError(f"Option 'even_test' with value {v} is not an even number.")

        options = MyOptions()
        options['even_test'] = 2
        options['even_test'] = 4

        with self.assertRaises(Exception) as context:
            options['even_test'] = 3

        msg = "Option 'even_test' with value 3 is not an even number."
        self.assertTrue(msg in str(context.exception))

    def test_unnamed_args(self):
        class MyOptions(_OptionsBaseModel):
            test: int = Field(default=1)

        options = MyOptions()
        with self.assertRaises(Exception) as context:
            options['testx'] = 1

        # KeyError ends up with an extra set of quotes.
        msg = "Object has no attribute 'testx"
        self.assertTrue(msg in str(context.exception))

        with self.assertRaises(Exception) as context:
            options['testx']

        msg = "'MyOptions' object has no attribute 'testx'"
        self.assertTrue(msg in str(context.exception))

    def test_contains(self):
        class MyOptions(_OptionsBaseModel):
            test: int = Field(default=1)

        options = MyOptions()

        self.assertFalse('testx' in options)
        self.assertTrue('test' in options)

    def test_update(self):
        class MyOptions(_OptionsBaseModel):
            foo: str = Field(default=None, desc='Test integer value')
            bar: Union[float, int] = Field(default=False)

        options = MyOptions()
        options['foo'] = 'b'
        options['bar'] = 3.14

        self.assertEqual(options['foo'], 'b')
        self.assertAlmostEqual(options['bar'], 3.14)
        options.update({'foo': 'c', 'bar': 5})
        self.assertEqual(options['foo'], 'c')
        self.assertEqual(options['bar'], 5)

    def test_update_extra(self):
        class MyOptions(_OptionsBaseModel):
            foo: str = Field(default=None, desc='Test integer value')
            bar: Union[float, int] = Field(default=False)

        options = MyOptions()
        options['foo'] = 'b'
        options['bar'] = 3.14

        self.assertEqual(options['foo'], 'b')
        self.assertAlmostEqual(options['bar'], 3.14)
        with self.assertRaises(Exception) as context:
            options.update({'foo': 'c', 'bar': 5, 'test': 2})

        msg = "test\n  Object has no attribute 'test' [type=no_such_attribute, input_value=2, input_type=int]"
        self.assertTrue(msg in str(context.exception))

    def test_bounds(self):
        class MyOptions(_OptionsBaseModel):
            x: float = Field(default=1.0, ge=0.0, le=2.0)

        options = MyOptions()

        with self.assertRaises(ValueError) as context:
            options['x'] = 3.0

        msg = "x\n  Input should be less than or equal to 2 [type=less_than_equal, input_value=3.0, input_type=float]"
        self.assertTrue(msg in str(context.exception))

        with self.assertRaises(ValueError) as context:
            options['x'] = -3.0

        msg = "x\n  Input should be greater than or equal to 0 [type=greater_than_equal, input_value=-3.0, input_type=float]"
        self.assertTrue(msg in str(context.exception))

    def test_deprecated_option(self):
        class MyOptions(_OptionsBaseModel):
            test1: float = Field(default=1.0, deprecated='Option "test1" is deprecated.')

        options = MyOptions()

        msg =  'Option "test1" is deprecated.'

        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)

            # test double set
            #with assert_warning(OMDeprecationWarning, msg):
            options['test1'] = 2.
            # Should only generate warning first time
            with assert_no_warning(OMDeprecationWarning, msg):
                options['test1'] = 2.

            # Also test set and then get
            msg = 'Option "test2" is deprecated.'
            options.declare('test2', deprecation=msg)

            with assert_warning(OMDeprecationWarning, msg):
                options['test2'] = None
            # Should only generate warning first time
            with assert_no_warning(OMDeprecationWarning, msg):
                options['test2']


if __name__ == "__main__":
    unittest.main()
