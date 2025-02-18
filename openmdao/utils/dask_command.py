import os
import numpy as np
import yaml


from openmdao.utils.file_utils import _load_and_exec, _get_object_from_script
from openmdao.core.problem import Problem
from openmdao.utils.om_warnings import issue_warning

try:
    from dask.distributed import Client, Worker, WorkerPlugin, get_worker, as_completed
except ImportError:
    dask = None


def _dask_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao dask' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Name of output file.  By default, output goes to stdout.')
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name')


def _locate_problem(globals_dict, problem_name, fname):
    if problem_name is None:
        probs = [obj for obj in globals_dict.values() if isinstance(obj, Problem)]
        if len(probs) == 1:
            return probs[0]
        elif len(probs) == 0:
            issue_warning(f"No Problem found in file '{fname}'.")
        else:
            issue_warning(f"Multiple Problems found in file '{fname}'. Using the first one, "
                          f"Problem '{probs[0]._name}'.")
            return probs[0]
    else:
        for obj in globals_dict.values():
            if isinstance(obj, Problem) and obj._name == problem_name:
                return obj

        issue_warning(f"Problem '{problem_name}' not found in file '{fname}'.")
        return None


def get_plugin(fname, problem_name=None, *user_args):
    # The setup function will be executed once on each worker when it starts.
    class OpenMDAOSetupPlugin(WorkerPlugin):
        def setup(self, worker):
            # load and run the model script to give the worker a problem
            globals_dict = _load_and_exec(fname, user_args)
            prob = _locate_problem(globals_dict, problem_name, fname)
            if prob is None:
                worker.close()
                return
            responses = prob.model.get_responses()
            worker.prob = prob  # Store the problem in worker's state for later use.
            worker.responses = responses

    return OpenMDAOSetupPlugin()


def run_model(case):
    ident, samples = case
    # Use the worker's stored state
    worker = get_worker()
    _set_inputs(worker.prob, samples)

    worker.prob.run_model()
    return _collect_responses(worker.prob, ident, worker.responses)


def optimize_model(case):
    ident, samples = case
    # Use the worker's stored state
    worker = get_worker()
    _set_inputs(worker.prob, samples)

    worker.prob.run_driver()
    return _collect_responses(worker.prob, ident, worker.responses)


def _set_inputs(problem, samples):
    for var, meta in samples.items():
        val = meta['val']
        units = meta.get('units', None)
        idxs = meta.get('indices', None)
        problem.set_val(var, val, units, idxs)


def _collect_responses(problem, ident, responses):
    # values must be copied to avoid race condition where later cases overwrite previous values
    return (ident, [np.copy(problem.get_val(name)) for name in responses])


def load_inputs(input_fname, input_objname=None):
    if input_fname.endswith('.py'):
        return _read_py(input_fname, input_objname)
    elif input_fname.endswith('.yaml'):
        return _read_yaml(input_fname)
    else:
        raise ValueError(f"Unrecognized file extension for file '{input_fname}'.")


def _read_py(input_fname, input_objname):
    return _get_object_from_script(input_fname, input_objname)


def _read_yaml(input_fname):
    with open(input_fname, "r") as yamlfile:
        return yaml.safe_load(yamlfile)


def run_dask(model_fname, input_fname, input_objname=None, num_workers=None, problem_name=None,
             do_opt=False, user_args=()):
    """
    Run a model with different inputs in parallel using Dask.

    Parameters
    ----------
    model_fname : str
        File name of the model script.
    input_fname : str
        File where the input data is stored. Data format is determined by the file extension, e.g.,
        .py, .yaml, etc.
    input_objname : str, optional
        Name of the object in the file that contains the input data. Only applicable if
        input_fname refers to a Python file.
    num_workers : int, optional
        Number of workers to use.  If not specified, the number of workers will be
        determined by the number of available cores.
    problem_name : str, optional
        Name of the problem to run.
    do_opt : bool, optional
        If True, run the optimizer for each case.
    user_args : list of str, optional
        Args to be passed to the user script.
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    inputs = load_inputs(input_fname, input_objname)
    # for inp in inputs:
    #     print(inp)

    # import sys
    # sys.exit()

    plugin = get_plugin(model_fname, problem_name, *user_args)

    # Start Dask client
    client = Client(n_workers=num_workers, threads_per_worker=1, worker_class=Worker)

    # Register the worker plugin so that each worker runs the setup code.
    client.register_plugin(plugin, name="openmdao_setup")

    func = optimize_model if do_opt else run_model

    print('inputs:')

    futures = []
    for i, idict in enumerate(inputs):
        print(i, idict.items())
        futures.append(client.submit(func, (i, idict), pure=False))

    print('results:')
    for future in as_completed(futures):
        print(future.result())
