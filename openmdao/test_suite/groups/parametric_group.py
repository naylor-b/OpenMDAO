"""Define the test group classes."""
from pydantic import Field, ConfigDict

from openmdao.core.group import Group, GroupModel, GroupOptions
from openmdao.utils.validation import DataModelManager as dmm


class ParametericTestGroup(Group):
    """
    Test Group expected by `ParametricInstance`. Groups inheriting from this should extend
    `default_params` to include valid parametric options for that model.

    Attributes
    ----------
    expected_totals : dict or None
        Dictionary mapping (out, in) pairs to the associated total derivative. Optional
    total_of : iterable
        Iterable containing which outputs to take the derivative of.
    total_wrt : iterable
        Iterable containing which variables with which to take the derivative of the above.
    expected_values : dict or None
        Dictionary mapping variable names to expected values. Optional.
    default_params : dict
        Dictionary containing the available options and default values for parametric sweeps.
    """
    def __init__(self, **kwargs):

        self.expected_totals = None
        self.total_of = None
        self.total_wrt = None
        self.expected_values = None
        self.default_params = {
            'local_vector_class': ['default', 'petsc'],
            'assembled_jac': [True, False],
            'jacobian_type': ['matvec', 'dense', 'sparse-csc'],
        }

        super().__init__(**kwargs)


        #self.options.update(kwargs)


class ParametericTestGroupOptions(GroupOptions):
    local_vector_class: str = Field(default='default', desc='Which local vector implementation to use.')
    assembled_jac: bool = Field(default=True, desc='If an assemebled Jacobian should be used.')
    jacobian_type: str = Field(default='matvec', desc='Controls the type of the assembled jacobian.')


@dmm.register(ParametericTestGroup)
class ParametericTestGroupModel(GroupModel):
    options: ParametericTestGroupOptions = Field(default_factory=ParametericTestGroupOptions)
