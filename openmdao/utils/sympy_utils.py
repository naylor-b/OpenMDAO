from ast import NodeTransformer, Subscript, Name, Load, Constant

import openmdao.func_api as omf


class RewriteName(NodeTransformer):

    def __init__(self, func):
        super().__init__()
        self._fwrap = omf.wrap(func)
        self._in_shapes = {n: shape for n, shape in self._fwrap.input_shape_iter()}
        self._out_shapes = {n: shape for n, shape in self._fwrap.output_shape_iter()}

    def visit_Name(self, node):
        return Subscript(
            value=Name(id='data', ctx=Load()),
            slice=Constant(value=node.id),
            ctx=node.ctx
        )

    def visit_Assign(self, node):
        for i,t in enumerate(node.targets):
            if i>0: self.append(',')
            self.visit(t)
        self.append(' = ')
        self.visit(node.value)

    def visit_Call(self, node):
        self.visit(node.func)
        self.append('(')
        total_args = 0
        for arg in node.args:
            if total_args>0: self.append(',')
            self.visit(arg)
            total_args += 1

        if hasattr(node, 'keywords'):
            for kw in node.keywords:
                if total_args>0: self.append(',')
                self.visit(kw)
                total_args += 1

        if hasattr(node, 'starargs'):
            if node.starargs:
                if total_args>0: self.append(',')
                self.append('*%s'%node.starargs)
                total_args += 1

        if hasattr(node, 'kwargs'):
            if node.kwargs:
                if total_args>0: self.append(',')
                self.append('**%s'%node.kwargs)

        self.append(')')
