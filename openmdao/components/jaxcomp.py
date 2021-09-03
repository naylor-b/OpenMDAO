"""Define the LinearSystemComp class."""

# import jax
from jax import jvp, vjp, vmap, random, jit, make_jaxpr
# import jax.numpy as jnp
import numpy as np
import inspect
import ast
import textwrap
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.code_utils import _ReturnNamesCollector
from openmdao.utils.om_warnings import issue_warning


def _get_annotations(func):
    """
    Retrieve annotation data for function inputs and return values.

    Parameters
    ----------
    func : function
        The function object.

    Returns
    -------
    dict
        Input metadata dictionary.
    dict
        Return value metadata dictionary.
    """
    annotations = getattr(func, '__annotations__', None)
    inmeta = {}
    outmeta = {}
    if annotations is not None:
        ret = None
        # get input info
        for name, meta in annotations.items():
            if name == 'return':
                ret = meta
            else:
                inmeta[name] = meta

        if ret is not None:  # we have output annotations
            for name, meta in ret:
                outmeta[name] = meta

    return inmeta, outmeta


def _get_outnames_from_code(func):
    """
    Use ast to determine (if possible) the names of the given function's return values.

    For example, if the given function has a line `return a, b, c`, then this will return
    ['a', 'b', 'c'].  If the function instead returns something other than simple names,
    those will be replaced with None.  For example, a function containing the line
    `return a, b+1, sin(c)` would cause this to return ['a', None, None].

    If the given function contains more than one return line, this will raise an exception.

    If an ast cannot be constructed for the function, if for example it is a wrapper for
    compiled code, an exception will be raised.

    Parameters
    ----------
    func : function
        A reference to the function whose source code will be parsed for return value names.
    """
    src = textwrap.dedent(inspect.getsource(func))
    scanner = _ReturnNamesCollector()
    scanner.visit(ast.parse(src, mode='exec'))
    return scanner._ret_names


def get_func_info(func):
    """
    Retrieve metadata associated with function inputs and return values.

    Return value metadata can come from annotations or (shape only) can be determined
    using jax if the input shapes or values are known.  Return value names can be defined
    in annotations or can be determined from the function itself provided that the return
    values are internal function variable names.

    Parameters
    ----------
    func : function
        The function to be queried for input and return value info.

    Returns
    -------
    dict
        Dictionary of metdata for inputs.
    dict
        Dictionary of metadata for return values.
    """
    # TODO: get func source and re-exec with redifined globals to replace numpy with jax numpy so
    # functions defined with regular numpy stuff internally will still work.
    ins = {}

    # first, retrieve inputs from the function signature
    params = inspect.signature(func).parameters
    for name, p in params.items():
        ins[name] = meta = {}
        meta['val'] = p.default if p.default is not inspect._empty else None
        if meta['val'] is not None:
            if np.isscalar(meta['val']):
                meta['shape'] = 1
            else:
                meta['shape'] = meta['val'].shape

    # grab any annotations
    inmeta, outmeta = _get_annotations(func)
    for name, meta in inmeta.items():
        if name in ins:
            m = ins[name]
            m.update(meta)
        else:
            ins[name] = meta

    # Parse the function code to possibly identify the names of the return values.
    # This only succeeds if simple varible names are returned, e.g.,   return a, b, c
    outlist = []
    try:
        onames = _get_outnames_from_code(func)
    except RuntimeError:
        #  this could happen if function is compiled or has multiple return lines
        issue_warning("Couldn't determine function return names based on AST.",
                      category=DistributedComponentWarning)
    else:
        for o in onames:
            if '.' in o:  # don't allow dots in return value names
                o = None
            outlist.append([o, {}])

    notfound = []
    for i, (oname, ometa) in enumerate(outmeta.items()):
        for tup in outlist:
            n, meta = tup
            if n == oname:
                meta.update(ometa)
                break
        else:  # didn't find oname
            notfound.append(oname)

    if notfound:  # try to fill in the unnamed slots with annotated output data
        inones = [i for i, (n, m) in enumerate(outlist) if n is None]  # indices with name of None
        if len(notfound) != len(inones):
            raise RuntimeError(f"Number of unnamed return values ({len(inones)}) doesn't match "
                               f"number of unmatched annotated return values ({len(notfound)}).")

    outs = {n: m for n, m in outlist}

    need_shape = not outmeta
    for ometa in outs.values():
        if ometa.get('shape') is None:
            need_shape = True
            break

    args = []
    for name, meta in ins.items():
        if meta['val'] is not None:
            args.append(meta['val'])
        else:
            shp = meta.get('shape')
            if shp is None:
                raise RuntimeError(f"Can't determine shape of input '{name}'.")
            elif need_shape:
                args.append(ShapedArray(shp, dtype=np.float32))

    if need_shape:  # output shapes weren't provided in annotations
        jxpr = make_jaxpr(func)
        v = jxpr(*args)
        for val, name in zip(v.out_avals, outs):
            outs[name]['shape'] = val.shape

    return ins, outs


_allowed_add_input_args = {
    'val', 'shape', 'src_indices', 'flat_src_indices', 'units', 'desc', 'tags', 'shape_by_conn',
    'copy_shape', 'distributed', 'new_style_idx',
}

_allowed_add_output_args = {
    'val', 'shape', 'units', 'res_units', 'desc' 'lower', 'upper', 'ref', 'ref0', 'res_ref', 'tags',
    'shape_by_conn', 'copy_shape', 'distributed',
}


class ADExplicitFuncComp(ExplicitComponent):
    """
    A component based on an annotated function.

    This component uses AD (jax) to compute its derivatives.

    Parameters
    ----------
    func : function
        The function to be wrapped by this Component.
    **kwargs : named args
        Args passed down to ExplicitComponent.

    Attributes
    ----------
    _func : function
        The function wrapped by this component.
    _inmeta : dict
        Function input metadata.
    _outmeta : dict
        Function return value metadata.
    """

    def __init__(self, func, **kwargs):
        """
        Initialize attributes.
        """
        super().__init__(**kwargs)
        self._func = func
        self._inmeta = None
        self._outmeta = None
        # self._jacfwd = None
        # self._jacrev = None

    def setup(self):
        """
        Define out inputs and outputs.
        """
        self._inmeta, self._outmeta = get_func_info(self._func)
        for name, meta in self._inmeta.items():
            kwargs = {n: v for n, v in meta.items() if n in _allowed_add_input_args}
            self.add_input(name, **kwargs)

        for name, meta in self._outmeta.items():
            kwargs = {n: v for n, v in meta.items() if n in _allowed_add_output_args}
            self.add_output(name, **kwargs)

    def setup_partials(self):
        """
        Set up our partial derivative sparsity.
        """
        pass

    def compute(self, inputs, outputs):
        """
        Compute the result of calling our function with the given inputs.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables.
        outputs : Vector
            Unscaled, dimensional output variables.
        """
        print("args:", list(inputs.values()))
        outs = self._func(*inputs.values())
        print("outs:", outs)
        arr = outputs.asarray()
        start = end = 0
        for o in outs:
            end += o.size
            arr[start:end] = np.asarray(o).flat
            start = end

        print("arr:", arr)
        print("outputs:", outputs.asarray())
        print("_outputs:", self._outputs.asarray())
        print("outputs['baz']", self._outputs['baz'])


if __name__ == '__main__':
    @jit
    def _some_func(x=np.zeros(4), y=np.ones(4), z=3):
        foo = 2. * x + 3. * y
        bar = 2 * (x + y)
        baz = z * 3. + 1.
        return foo, bar, baz

    import openmdao.api as om

    p = om.Problem()
    p.model.add_subsystem('comp', ADExplicitFuncComp(_some_func))
    p.setup()
    p['comp.x'] = np.arange(4, dtype=float) + 1.
    p['comp.y'] = np.arange(4, dtype=float) + 2.
    p['comp.z'] = 99.

    print('comp.foo', p['comp.foo'])
    print('comp.bar', p['comp.bar'])
    print('comp.baz', p['comp.baz'])

    print("run model")
    p.run_model()

    print('comp.foo', p['comp.foo'])
    print('comp.bar', p['comp.bar'])
    print('comp.baz', p['comp.baz'])
