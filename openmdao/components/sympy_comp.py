import sys
import textwrap
import inspect
import ast
from types import FunctionType
import sympy
from sympy import derive_by_array, diff, Symbol, Matrix, Array, MatrixSymbol, cse, \
    flatten, zeros
from sympy.core.function import FunctionClass
import numpy as np
from astunparse import unparse

import openmdao.func_api as omf
from openmdao.core.constants import _UNDEFINED
from openmdao.components.func_comp_common import namecheck_rgx, _disallowed_varnames
from openmdao.utils.sympy_utils import SymArrayTransformer


def _check_var_name(name):
    match = namecheck_rgx.match(name)
    if match is None or match.group() != name:
        raise NameError(f"'{name}' is not a valid variable name.")

    if name in _disallowed_varnames:
        raise NameError(f"Can't use variable name '{name}' because "
                        "it's a reserved keyword.")


def _arr_fnc_(func, expr):
    if isinstance(expr, Array):
        return expr.applyfunc(func)
    return func(expr)


def _do_mult_(left, right):
    if isinstance(left, Array) and isinstance(right, Array):
        if left.shape != right.shape:
            raise RuntimeError(f"Tried to multiply array of shape {left.shape} by array of "
                                f"shape {right.shape}.")
        return Array([a * b for a, b in zip(flatten(left), flatten(right))]).reshape(left.shape)
    return left * right


# def _dodiv(left, right):
#     if isinstance(left, Array) and isinstance(right, Array):
#         if left.shape != right.shape:
#             raise RuntimeError(f"Tried to divide array of shape {left.shape} by array of "
#                                 f"shape {right.shape}.")
#         return Array([a / b for a, b in zip(flatten(left), flatten(right))]).reshape(left.shape)
#     return left / right


def _gen_setup(fwrapper):
    """
    Define out inputs, outputs, and options
    """
    optignore = {'is_option'}

    lines = ['    def setup(self):']

    for name, meta in fwrapper.get_input_meta():
        _check_var_name(name)
        if 'is_option' in meta and meta['is_option']:
            kwargs = {k: v for k, v in meta.items() if k in omf._allowed_declare_options_args and
                      k not in optignore}
            opts = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            lines.append(f"        self.options.declare('{name}', {opts})")
        else:
            kwargs = omf._filter_dict(meta, omf._allowed_add_input_args)
            optlist = []
            for k, v in kwargs.items():
                if k == 'val':
                    if isinstance(v, np.ndarray):
                        shp = v.shape
                        if np.all(v == np.ones(shp)):
                            optlist.append(f'{k}=np.ones({shp})')
                        elif np.all(v == np.zeros(shp)):
                            optlist.append(f'{k}=np.zeros({shp})')
                        else:
                            optlist.append(f'{k}={v}'.replace('\n', ''))
                else:
                    optlist.append(f'{k}={v}')

            lines.append(f"        self.add_input('{name}', {', '.join(optlist)})")

    for name, meta in fwrapper.get_output_meta():
        _check_var_name(name)
        kwargs = {k: v for k, v in meta.items() if k in omf._allowed_add_output_args and
                  k != 'resid'}
        optlist = []
        for k, v in kwargs.items():
            if k == 'val':
                if isinstance(v, np.ndarray):
                    shp = v.shape
                    if np.all(v == np.ones(shp)):
                        optlist.append(f'{k}=np.ones({shp})')
                    elif np.all(v == np.zeros(shp)):
                        optlist.append(f'{k}=np.zeros({shp})')
                    else:
                        optlist.append(f'{k}={v}'.replace('\n', ''))
            else:
                optlist.append(f'{k}={v}')

        lines.append(f"        self.add_output('{name}', {', '.join(optlist)})")

    return '\n'.join(lines)


def _gen_setup_partials(partials):
    lines = ['    def setup_partials(self):']
    for of, wrt in partials:
        lines.append(f"        self.declare_partials(of='{of}', wrt='{wrt}')")

    return '\n'.join(lines)


def _gen_compute_partials(inputs, partials, replacements, reduced_exprs):
    inames = ', '.join(inputs)
    lines = [
        '    def compute_partials(self, inputs, partials):',
        f'        {inames} = inputs.values()'
    ]

    for name, val in replacements:
        lines.append(f"        {name} = {val}" )

    for i, key in enumerate(partials):
        of, wrt = key
        lines.append(f"        partials['{of}', '{wrt}'] = {reduced_exprs[i]}")

    return '\n'.join(lines)


def _get_sparsity(J):
    nrows, ncols = J.shape
    if nrows == ncols and J.is_diagonal():
        idxs = list(range(nrows))
        return idxs, idxs
    arr = np.zeros(J.shape, dtype=bool)
    for r in range(nrows):
        for c in range(ncols):
            if J[r, c] != 0:
                arr[r, c] = True
    return np.nonzero(arr)


def get_symbolic_derivs(fwrap, optimizations='basic'):
    func = fwrap._f
    inputs = list(fwrap.get_input_names())
    outputs = list(fwrap.get_output_names())

    # use sympy module as globals so we get symbolic versions of common functions
    globdict = sympy.__dict__.copy()

    # create src that declares the function and then calls it, returning the output values.
    # We inject sympy Symbols or Arrays in place of input arguments so the outputs will be
    # the appropriate symbolic objects.  Similar to lambdify but with the addition of Arrays
    # so we get correct jacobians (and determine their sparsity easily).
    funcsrc = textwrap.dedent(inspect.getsource(func))
    print("original func:")
    print(funcsrc)

    tree = ast.parse(funcsrc)

    functs2convert = {n for n, f in globdict.items()
                      if isinstance(f, FunctionClass) and len(inspect.signature(f).parameters) == 1}

    # add conversion functs to globals
    globdict['_arr_fnc_'] = _arr_fnc_
    globdict['_do_mult_'] = _do_mult_

    xform = SymArrayTransformer(functs2convert)

    node = ast.fix_missing_locations(xform.visit(tree))

    # get the source for the converted function ast
    funcsrc = unparse(node)
    print("converted func:")
    print(funcsrc)

    callfunc = f"{', '.join(outputs)} = {func.__name__}({', '.join(inputs)})"
    src = '\n'.join([funcsrc, callfunc])

    code = compile(src, mode='exec', filename='<string>')

    # inject symbolic inputs into the locals dict
    locdict = {}
    for name, shape in fwrap.input_shape_iter():
        if shape == ():
            locdict[name] = Symbol(name)
        else:
            locdict[name] = Array(flatten(MatrixSymbol(name, *shape)), shape)

    exec(code, globdict, locdict)  # nosec: limited to _expr_dict

    partials = {}
    for out in outputs:
        for inp in inputs:
            # J = diff(locdict[out], locdict[inp])
            # J = derive_by_array(locdict[out], locdict[inp])
            # print(type(locdict[out]))
            # print("OF:", Matrix(flatten(locdict[out])))
            J = Matrix(flatten(locdict[out])).jacobian(flatten(locdict[inp]))
            if not J.equals(zeros(*J.shape)):
                sparsity = _get_sparsity(J)
                print("sparsity:", sparsity)
                print("diag:", J.is_diagonal())
                # TODO: determine here if J is linear (not a symbol or expression)
                partials[(out, inp)] = J

    replacements, reduced_exprs = cse(partials.values(), order=None, optimizations=optimizations)

    return partials, replacements, reduced_exprs


def gen_sympy_explicit_comp(func, classname, out_stream=_UNDEFINED):
    fwrap = omf.wrap(func)
    if isinstance(func, omf.OMWrappedFunc):
        func = fwrap._f  # get the original function

    inputs = list(fwrap.get_input_names())
    outputs = list(fwrap.get_output_names())
    partials, replacements, reduced_exprs = get_symbolic_derivs(fwrap)

    # sparsity = _get_sparsity(fwrap, partials, replacements, reduced_exprs)

    # generate setup method
    setup_str = _gen_setup(fwrap)
    # TODO: generate sparsity info
    setup_partials_str = _gen_setup_partials(partials)
    # generate compute_partials method
    compute_partials_str = _gen_compute_partials(inputs, partials,
                                                 replacements, reduced_exprs)

    funcsrc = textwrap.dedent(inspect.getsource(func))

    src = f"""
from openmdao.core.explicitcomponent import ExplicitComponent

{funcsrc}

class {classname}(ExplicitComponent):
{setup_str}

{setup_partials_str}

    def compute(self, inputs, outputs):
        outputs.set_vals({func.__name__}(*inputs.values()))

{compute_partials_str}

if __name__ == '__main__':
    import openmdao.api as om
    import numpy as np
    from numpy import *
    p = om.Problem()
    comp = p.model.add_subsystem('comp', {classname}(), promotes=['*'])
    comp.declare_coloring()
    p.setup()
    p.run_model()
    J = p.compute_totals(of={inputs}, wrt={outputs})
    print(J)
    """

    if out_stream is _UNDEFINED:
        out_stream = sys.stdout

    if out_stream is not None:
        print(src, file=out_stream)

    return src


if __name__ == '__main__':
    import openmdao.api as om
    from math import log2, sin, cos

    # def myfunc(a, b, c):
    #     x = a**2 * sin(b) + cos(c)
    #     y = sin(a + b + c)
    #     z = log(a*b)
    #     return x, y, z

    def myfunc(a, b, c):
        x = cos(a)
        y = sin(b)
        z = sin(c) + cos(c)
        return x, y, z

    fwrap = (omf.wrap(myfunc)
               .defaults(shape=(3,2)))

    with open('_mysympycomp.py', 'w') as f:
        fstr = gen_sympy_explicit_comp(fwrap, 'MySympyComp', f)

    print(fstr)

    # exec(fstr)

    # comp = globals()['MySympyComp']()
    # comp.declare_coloring()

    # p = om.Problem()
    # p.model.add_subsystem('comp', comp, promotes=['*'])
    # p.setup()
    # p.run_model()

    # J = p.compute_totals(of=list(fwrap.get_output_names()), wrt=list(fwrap.get_input_names()))
    # print(J)
