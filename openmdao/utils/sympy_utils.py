from ast import NodeTransformer, Name, Mult, Call
from sympy import *


# def _do_mult_(a, b):
#     if isinstance(a, Array) and isinstance(b, Array):
#         return [v1 * v2 for v1, v2 in zip(flatten(a), flatten(b))]
#     return a * b


# def _arr_fnc_(fnc, *args):
#     if len(args) == 1 and isinstance(args[0], Array):
#         return args[0].applyfunc(fnc)
#     return fnc(*args)



class SymArrayTransformer(NodeTransformer):
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
            return Call(Name(id=self._xfname), newargs, node.keywords)

        node.args = newargs
        return node

    def visit_BinOp(self, node):  # left, op, right
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.op, Mult):
            args = [nl, nr]
            return Call(Name(id=self._multfname), args, [])

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
