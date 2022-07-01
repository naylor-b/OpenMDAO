
import sys
import ast
from ast import Name, Mult, Call
from sympy import *
import inspect
import textwrap
if sys.version_info >= (3, 9):
    from ast import unparse
else:
    try:
        from astunparse import unparse
    except ImportError:
        unparse = None


class SymArrayTransformer(ast.NodeTransformer):
    def __init__(self, funcs, xfname='_arr_fnc_', multfname='_do_mult_'):
        super().__init__()
        self._transform_funcs = funcs
        self._xfname = xfname
        self._multfname = multfname

    def visit_Call(self, node):  # func, args, keywords
        f = self.visit(node.func)
        newargs = [self.visit(a) for a in node.args]
        for kwd in node.keywords:
            kwd.value = self.visit(kwd.value)
        if isinstance(node.func, Name) and node.func.id in self._transform_funcs:
            newargs = [f] + newargs
            return Call(Name(id=self._xfname, ctx=ast.Load()), newargs, node.keywords)

        node.args = newargs
        return node

    def visit_BinOp(self, node):  # left, op, right
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.op, Mult):
            args = [node.left , node.right]
            return Call(Name(id=self._multfname, ctx=ast.Load()), args, [])

        return node


def _arr_fnc_(func, expr):
    if isinstance(expr, (Array, Matrix)):
        return expr.applyfunc(func)
    return func(expr)


def _do_mult_(left, right):
    if isinstance(left, Array) and isinstance(right, Array):
        lshape = left.shape
        if lshape != right.shape:
            raise RuntimeError(f"Tried to multiply array of shape {lshape} by array of "
                                f"shape {right.shape}.")
        if lshape != ():
            left = flatten(left)
            right = flatten(right)
        return Array([a * b for a, b in zip(left, right)]).reshape(*lshape)
    return left * right


def func2sympy_compat(fwrap, global_dict, show=False):
    """
    Convert any multiplications or function applications to calls to _do_mult_ or _arr_func.

    This allows us to replace some func args with Arrays and still be able to execute the function.

    Returns
    -------
    ast node
        An ast node corresponding to the top of the ast that defines the modified function.
    """
    functs2convert = {n for n, f in global_dict.items()
                      if isinstance(f, FunctionClass) and len(inspect.signature(f).parameters) == 1}

    # add conversion functs to globals
    global_dict['_arr_fnc_'] = _arr_fnc_  # applies a func to a whole Array
    global_dict['_do_mult_'] = _do_mult_  # treat Array mult as in-place entry mult

    xform = SymArrayTransformer(functs2convert)

    func = fwrap._f
    funcsrc = textwrap.dedent(inspect.getsource(func))

    # add a call to the function in the source so when we exec it we'll get
    # the output(s) of the function which we'll use to compute derivatives.
    inputs = list(fwrap.get_input_names())
    outputs = list(fwrap.get_output_names())
    callfunc = f"{', '.join(outputs)} = {func.__name__}({', '.join(inputs)})"
    src = '\n'.join([funcsrc, callfunc])

    tree = ast.parse(src)

    node = ast.fix_missing_locations(xform.visit(tree))

    if show and unparse is not None:
        # print the source for the converted function ast
        print(unparse(node))

    return node


if __name__ == '__main__':
    import ast
    from sympy import sin, cos, tan
    import inspect
    import textwrap
    from astunparse import unparse

    def myfunc(a, b, c):
        x = cos(a) * tan(b)
        y = sin(b) * 3.
        z = sin(c) - cos(a)
        return x, y, z

    src = textwrap.dedent(inspect.getsource(myfunc))
    tree = ast.parse(src)

    xff = {'sin', 'cos'}
    xform = SymArrayTransformer(xff)

    node = ast.fix_missing_locations(xform.visit(tree))

    print(unparse(node))
