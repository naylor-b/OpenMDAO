import textwrap
import inspect
import sympy
from sympy import sympify, diff, symbols

import openmdao.func_api as omf
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


def _gen_compute_partials(inputs, partials):
    innames = ', '.join(inputs)
    lines = [
        '    def compute_partials(self, inputs, partials):',
        f'        {innames} = inputs.values()'
    ]

    for key, partial in partials.items():
        of, wrt = key
        lines.append(f"        partials['{of}', '{wrt}'] = {partial}")

    return '\n'.join(lines)


def get_symbolic_derivs(fwrap):
    func = fwrap._f
    inputs = list(fwrap.get_input_names())
    outputs = list(fwrap.get_output_names())

    print("return names are:", fwrap.get_return_names())
    print("input names:", inputs)
    print("output names:", outputs)

    symarg = " ".join(inputs)
    funcsrc = textwrap.dedent(inspect.getsource(func))
    declsyms = f"\n{', '.join(inputs)} = symbols('{symarg}')"
    callfunc = f"{', '.join(outputs)} = {func.__name__}({', '.join(inputs)})"
    src = '\n'.join([funcsrc, declsyms, callfunc])

    # print(src)

    code = compile(src, mode='exec', filename='<string>')

    locdict = {}
    for name, sym in zip(inputs, symbols(' '.join(inputs))):
        locdict[name] = sym

    globdict = sympy.__dict__.copy()
    exec(code, globdict, locdict)  # nosec: limited to _expr_dict

    # import pprint
    # pprint.pprint(locdict)

    partials = {}
    for out in outputs:
        for inp in inputs:
            dff = diff(locdict[out], locdict[inp])
            print(f"d{out}/d{inp} = {dff}")
            if dff != 0:
                partials[(out, inp)] = dff

    return partials

    # def setup_partials(self):
    #     """
    #     Set up the derivatives.
    #     """
    #     vec_size = self.options['vec_size']
    #     vec_size_A = self.vec_size_A = vec_size if self.options['vectorize_A'] else 1
    #     size = self.options['size']
    #     mat_size = size * size
    #     full_size = size * vec_size

    #     row_col = np.arange(full_size, dtype="int")

    #     self.declare_partials('x', 'b', val=np.full(full_size, -1.0), rows=row_col, cols=row_col)

    #     rows = np.repeat(np.arange(full_size), size)

    #     if vec_size_A > 1:
    #         cols = np.arange(mat_size * vec_size)
    #     else:
    #         cols = np.tile(np.arange(mat_size), vec_size)

    #     self.declare_partials('x', 'A', rows=rows, cols=cols)

    #     cols = np.tile(np.arange(size), size)
    #     cols = np.tile(cols, vec_size) + np.repeat(np.arange(vec_size), mat_size) * size

    #     self.declare_partials(of='x', wrt='x', rows=rows, cols=cols)


def gen_sympy_comp(func):
    fwrap = omf.wrap(func)
    partials = get_symbolic_derivs(fwrap)

    # generate setup method
    print(_gen_setup(fwrap))
    # TODO: generate sparsity info
    print(_gen_setup_partials(partials))
    # generate compute_partials method
    print(_gen_compute_partials(fwrap.get_input_names(), partials))


if __name__ == '__main__':
    from math import log2, sin, cos

    def myfunc(a, b, c):
        x = a**2 * sin(b) + cos(c)
        y = sin(a + b + c)
        z = log(a*b)
        return x, y, z

    gen_sympy_comp(myfunc)

