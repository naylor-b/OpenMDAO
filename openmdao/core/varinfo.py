
import numpy as np
from numpy import reshape, isscalar, ndim, full

from openmdao.utils.class_util import auto_forward
from openmdao.utils.array_utils import shape_to_len
from openmdao.utils.general_utils import make_set, shape2tuple, _valid_var_name
from openmdao.utils.code_utils import is_lambda, LambdaPickleWrapper
from openmdao.utils.units import simplify_unit
from openmdao.visualization.tables.table_builder import generate_table


flag_type = np.uint8

base = flag_type(1)

DISCRETE = base << 0
REMOTE = base << 1
DISTRIBUTED = base << 2
REQUIRE_CONNECTION = base << 3
SHAPE_BY_CONN = base << 4
UNITS_BY_CONN = base << 5
IS_INPUT = base << 6

BY_CONN = SHAPE_BY_CONN | UNITS_BY_CONN


class VarInfo():
    """
    Variable information that is fixed across all processes and systems.
    """
    __slots__ = ('name', '_tags', 'desc', '_flags')

    def __init__(self, name, io):
        self._flags = flag_type(0)

        if not isinstance(name, str):
            raise TypeError('The name argument must be a string.')
        if not _valid_var_name(name):
            raise NameError(f"'{name}' is not a valid variable name.")

        self.name = name
        if io == 'input':
            self.is_input = True
        elif io != 'output':
            raise ValueError(f"Invalid io type: '{io}'")

        self._tags = set()

    def __getattr__(self, key):
        return None

    # for backward compatibility, add __getitem__ and __setitem__
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def update(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def iostr(self):
        return 'input' if self._flags & IS_INPUT else 'output'

    @property
    def is_input(self):
        return bool(self._flags & IS_INPUT)

    @is_input.setter
    def is_input(self, value):
        if value:
            self._flags |= IS_INPUT
        else:
            self._flags &= ~IS_INPUT

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, value):
        self._tags = make_set(value)

    @property
    def discrete(self):
        return bool(self._flags & DISCRETE)

    @discrete.setter
    def discrete(self, value):
        if value:
            self._flags |= DISCRETE
        else:
            self._flags &= ~DISCRETE

    @property
    def remote(self):
        return bool(self._flags & REMOTE)

    @remote.setter
    def remote(self, value):
        if value:
            self._flags |= REMOTE
        else:
            self._flags &= ~REMOTE

    @property
    def require_connection(self):
        return bool(self._flags & REQUIRE_CONNECTION)

    @require_connection.setter
    def require_connection(self, value):
        if self.is_input:
            if value:
                self._flags |= REQUIRE_CONNECTION
            else:
                self._flags &= ~REQUIRE_CONNECTION
        else:
            raise ValueError("'require_connection' cannot be set on an output variable.")


class ContinuousVarInfo(VarInfo):
    __slots__ = ('_shape', '_global_shape', '_units', '_copy_shape',
                 '_compute_shape', '_copy_units', '_compute_units')

    def __init__(self, name, io, **kwargs):
        super().__init__(name, io)
        self._shape = None
        self._global_shape = None
        self._units = None
        self._copy_shape = None
        self._compute_shape = None
        self._copy_units = None
        self._compute_units = None
        self.update(kwargs)

    def __repr__(self):
        rows = [[key, getattr(self, key)] for key in self.__slots__
                if getattr(self, key) is not None and key != 'meta']
        table = generate_table(rows, tablefmt='plain')
        return f"{self.__class__.__name__}:\n{table}"

    @property
    def distributed(self):
        return bool(self._flags & DISTRIBUTED)

    @distributed.setter
    def distributed(self, value):
        if value:
            self._flags |= DISTRIBUTED
        else:
            self._flags &= ~DISTRIBUTED

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape2tuple(shape)

    @property
    def size(self):
        return shape_to_len(self._shape)

    @property
    def global_size(self):
        return shape_to_len(self._global_shape)

    @property
    def shape_by_conn(self):
        return bool(self._flags & SHAPE_BY_CONN)

    @shape_by_conn.setter
    def shape_by_conn(self, value):
        if value:
            self._flags |= SHAPE_BY_CONN
        else:
            self._flags &= ~SHAPE_BY_CONN

    @property
    def copy_shape(self):
        return self._copy_shape

    @copy_shape.setter
    def copy_shape(self, value):
        if value is not None:
            if self._compute_shape is not None:
                raise ValueError("Only one of 'copy_shape' or 'compute_shape' can be specified.")
            if not isinstance(value, str):
                raise TypeError(f"The copy_shape argument should be a str or None but "
                                f"a '{type(value).__name__}' was given.")
        self._copy_shape = value

    @property
    def compute_shape(self):
        return self._compute_shape

    @compute_shape.setter
    def compute_shape(self, value):
        if value is not None:
            if self._copy_shape is not None:
                raise ValueError("Only one of 'copy_shape' or 'compute_shape' can be specified.")
            if not callable(value):
                raise TypeError(f"The compute_shape argument should be callable but "
                                f"a '{type(value).__name__}' was given.")
            if is_lambda(value):
                value = LambdaPickleWrapper(value)
        self._compute_shape = value

    @property
    def dyn_shape(self):
        return self.shape_by_conn or self.copy_shape is not None or self.compute_shape is not None

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if value is not None:
            self._units = simplify_unit(value)

    @property
    def units_by_conn(self):
        return bool(self._flags & UNITS_BY_CONN)

    @units_by_conn.setter
    def units_by_conn(self, value):
        if value:
            self._flags |= UNITS_BY_CONN
        else:
            self._flags &= ~UNITS_BY_CONN

    @property
    def copy_units(self):
        return self._copy_units

    @copy_units.setter
    def copy_units(self, value):
        if value is not None:
            if self._compute_units is not None:
                raise ValueError("Only one of 'copy_units' or 'compute_units' can be specified.")
            if not isinstance(value, str):
                raise TypeError(f"The copy_units argument should be a str or None but "
                                f"a '{type(value).__name__}' was given.")
        self._copy_units = value

    @property
    def compute_units(self):
        return self._compute_units

    @compute_units.setter
    def compute_units(self, value):
        if value is not None:
            if self._copy_units is not None:
                raise ValueError("Only one of 'copy_units' or 'compute_units' can be specified.")
            if not callable(value):
                    raise TypeError(f"The compute_units argument should be callable but "
                                    f"a '{type(value).__name__}' was given.")
            if is_lambda(value):
                value = LambdaPickleWrapper(value)
        self._compute_units = value

    @property
    def dyn_units(self):
        return self.units_by_conn or self.copy_units is not None or self.compute_units is not None

    @property
    def by_conn(self):
        return bool(self._flags & BY_CONN)

    @property
    def dynamic(self):
        return self.dyn_shape or self.dyn_units


class ContinuousInputVarInfo(ContinuousVarInfo):

    __slots__ = ('src_indices',)

    def __init__(self, name, io, **kwargs):
        super().__init__(name, io)
        self.src_indices = None
        self.update(kwargs)


class ContinuousOutputVarInfo(ContinuousVarInfo):
    __slots__ = ('_ref', '_ref0', '_res_ref', '_res_units', '_lower', '_upper')

    def __init__(self, name, io, **kwargs):
        super().__init__(name, io)
        # TODO: add properties with setters with checks for valid values
        self._ref = 1.0
        self._ref0 = 0.0
        self._res_ref = None
        self._res_units = None
        self._lower = None
        self._upper = None
        self.update(kwargs)


class ContinuousVariable():
    __slots__ = ('_varinfo', '_val')

    def __init__(self, varinfo):
        self._varinfo = varinfo
        self._val = None

    # for backward compatibility, add __getitem__ and __setitem__
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def update(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def shape(self):
        return self._varinfo._shape

    @shape.setter
    def shape(self, shape):
        self._varinfo._shape = shape
        if shape is not None:
            if shape != ():
                if self._val is not None and ndim(self._val) == 0:
                    # if val is a scalar, reshape it to the new shape
                    self._val = full(shape, self._val)

            if self._val is None:
                self._val = np.ones(shape)

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        if value is not None:
            if self._varinfo._shape is None:
                self._val = value
                # only set shape based on value if value is an array, because we must support
                # setting shape later on, expanding the scalar value into the given shape. It also
                # simplifies the connection graph if we don't have to worry about incorrect shapes
                # of (1,) or () propagating through a connection tree.
                if ndim(value) > 0:
                    self._varinfo._shape = np.shape(value)
            else:
                if self._varinfo._shape == ():
                    if isscalar(value):
                        self._val = value
                    else:
                        self._val = value.item()
                elif self._val is None:
                    self._val = reshape(value, self._shape)
                else:
                    self._val[:] = reshape(value, self._shape)


_varinfo_forward = ('name', 'desc', 'tags', 'discrete', 'remote', 'require_connection', 'is_input')
_continuous_varinfo_forward = ('distributed', 'shape', 'size', 'global_size', 'shape_by_conn',
                              'copy_shape', 'compute_shape', 'dyn_shape', 'units', 'units_by_conn',
                              'copy_units', 'compute_units', 'dyn_units', 'by_conn', 'dynamic')
_continuous_input_varinfo_forward = ('src_indices',)
_continuous_output_varinfo_forward = ('_ref', '_ref0', '_res_ref', '_res_units', '_lower', '_upper')

@auto_forward('_varinfo',
              _varinfo_forward + _continuous_varinfo_forward + _continuous_input_varinfo_forward)
class ContinuousInputVariable(ContinuousVariable):
    """
    Local continuous variable specific to the current process.
    """
    def __init__(self, name, io, **kwargs):
        super().__init__(ContinuousInputVarInfo(name, io))
        self.update(kwargs)


@auto_forward('_varinfo',
              _varinfo_forward + _continuous_varinfo_forward + _continuous_output_varinfo_forward)
class ContinuousOutputVariable(ContinuousVariable):
    """
    Local continuous variable specific to the current process.
    """
    def __init__(self, name, io, **kwargs):
        super().__init__(ContinuousOutputVarInfo(name, io))
        self.update(kwargs)


class DiscreteVarInfo(VarInfo):
    """
    Discrete variable information that is fixed across all processes.
    """
    def __init__(self, name, io, **kwargs):
        super().__init__(name, io)
        self.discrete = True
        self.update(kwargs)


@auto_forward('_varinfo', _varinfo_forward)
class DiscreteVariable():
    """
    Local discrete variable information specific to the current process.
    """
    __slots__ = ('_varinfo', 'val')

    def __init__(self, name, io):
        self._varinfo = DiscreteVarInfo(name, io)
        self._val = None


if __name__ == '__main__':
    outvar = ContinuousOutputVariable('y', 'output')
    outvar.val = 10.0
    print(outvar.val)
    print(outvar.discrete)