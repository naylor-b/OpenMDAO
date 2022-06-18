import sys
import textwrap
import inspect
import sympy
from sympy import diff, symbols, cse

import openmdao.func_api as omf
from openmdao.core.constants import _UNDEFINED
from openmdao.components.func_comp_common import namecheck_rgx, _disallowed_varnames


class MultiDict(object):
    """
    A dict wrapper that contains multiple dicts.

    Items are looked up in dicts in order.

    Attributes
    ----------
    _dicts : list
        List of dicts that will be searched for items.
    """

    def __init__(self, *dicts):
        """
        Create the dicts wrapper.

        Parameters
        ----------
        *dicts : dict-like positional args.
            Each arg is a dict-like object that may be searched.
        """
        self._dicts = dicts
        self._name2dict = {}

    def __getitem__(self, name):
        try:
            return self._name2dict[name][name]
        except KeyError:
            for d in self._dicts:
                if name in d:
                    self._name2dict[name] = d
                    return d[name]
            else:
                raise KeyError(f"Key '{name}' not found.")

    def __setitem__(self, name, value):
        try:
            self._name2dict[name][name] = value
        except KeyError:
            for d in self._dicts:
                if name in d:
                    self._name2dict[name] = d
                    d[name] = value
            else:
                raise KeyError(f"Key '{name}' not found.")

    def __contains__(self, name):
        if name in self._name2dict:
            return True
        for d in self._dicts:
            if name in d:
                self._name2dict[name] = d
                return True
        return False


def _check_var_name(name):
    match = namecheck_rgx.match(name)
    if match is None or match.group() != name:
        raise NameError(f"'{name}' is not a valid variable name.")

    if name in _disallowed_varnames:
        raise NameError(f"Can't use variable name '{name}' because "
                        "it's a reserved keyword.")


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
            opts = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
            lines.append(f"        self.add_input('{name}', {opts})")

    for name, meta in fwrapper.get_output_meta():
        _check_var_name(name)
        kwargs = {k: v for k, v in meta.items() if k in omf._allowed_add_output_args and
                    k != 'resid'}
        opts = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        lines.append(f"        self.add_output('{name}', {opts})")

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


def get_symbolic_derivs(fwrap):
    func = fwrap._f
    inputs = list(fwrap.get_input_names())
    outputs = list(fwrap.get_output_names())

    symarg = " ".join(inputs)
    funcsrc = textwrap.dedent(inspect.getsource(func))
    declsyms = f"\n{', '.join(inputs)} = symbols('{symarg}')"
    callfunc = f"{', '.join(outputs)} = {func.__name__}({', '.join(inputs)})"
    src = '\n'.join([funcsrc, declsyms, callfunc])

    code = compile(src, mode='exec', filename='<string>')

    locdict = {}
    for name, sym in zip(inputs, symbols(' '.join(inputs))):
        locdict[name] = sym

    globdict = sympy.__dict__.copy()
    exec(code, globdict, locdict)  # nosec: limited to _expr_dict

    partials = {}
    for out in outputs:
        for inp in inputs:
            dff = diff(locdict[out], locdict[inp])
            if dff != 0:
                # TODO: determine here if dff is linear (not a symbol or expression)
                partials[(out, inp)] = dff

    replacements, reduced_exprs = cse(partials.values(), order=None)

    return partials, replacements, reduced_exprs


def gen_sympy_explicit_comp(func, classname, out_stream=_UNDEFINED):
    fwrap = omf.wrap(func)
    partials, replacements, reduced_exprs = get_symbolic_derivs(fwrap)

    # generate setup method
    setup_str = _gen_setup(fwrap)
    # TODO: generate sparsity info
    setup_partials_str = _gen_setup_partials(partials)
    # generate compute_partials method
    compute_partials_str = _gen_compute_partials(fwrap.get_input_names(), partials,
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
    p = om.Problem()
    p.model.add_subsystem('comp', {classname}())
    """

    if out_stream is _UNDEFINED:
        out_stream = sys.stdout

    if out_stream is not None:
        print(src, file=out_stream)

    return str


if __name__ == '__main__':
    from math import log2, sin, cos

    def myfunc(a, b, c):
        x = a**2 * sin(b) + cos(c)
        y = sin(a + b + c)
        z = log(a*b)
        return x, y, z

    with open('mysympycomp.py', 'w') as f:
        gen_sympy_explicit_comp(myfunc, 'MySympyComp', f)
