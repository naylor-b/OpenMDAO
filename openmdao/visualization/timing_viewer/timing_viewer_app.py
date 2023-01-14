
import os
import time
import pickle
from functools import partial
import webbrowser
import threading
import json

import tornado.ioloop
import tornado.web

import openmdao.utils.hooks as hooks
from openmdao.core.problem import _problem_names, set_default_prob_name, num_problems
from openmdao.visualization.timing_viewer.timer import timing_context, _set_timer_setup_hook, \
    _save_timing_data, _main_table_row_iter, _global_info
import openmdao.visualization.timing_viewer.timer as timer_mod
from openmdao.utils.file_utils import _load_and_exec, _to_filename
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.mpi import MPI


_browsers = ['safari', 'chrome', 'firefox', 'chromium']


def launch_browser(port):
    time.sleep(1)
    for browser in _browsers:
        try:
            webbrowser.get(browser).open(f'http://localhost:{port}')
        except Exception:
            pass
        else:
            break
    else:
        print(f"Tried browsers {_browsers}, but all failed to launch.")


def start_thread(fn):
    thread = threading.Thread(target=fn)
    thread.setDaemon(True)
    thread.start()
    return thread


def shrink(s, wid=40):
    if len(s) <= wid:
        return s
    offset = (wid - 3) // 2
    return s[:offset] + '...' + s[-offset:]


def format_time(dt, digits=2):
    return f"{dt:.{digits}f}"


def ftup2key(ftup):
    # (rank, probname, sysname, method_name)
    return (ftup[0], ftup[1], ftup[3], ftup[7])


_view_options = [
    'text',
    'browser',
    # 'dump',
    'none'
]


class Application(tornado.web.Application):
    def __init__(self, db_fname):
        self.func_to_id = {}
        self.db_fname = db_fname

        self.total_time, self.nprocs = _global_info(db_fname)

        handlers = [
            (r"/", Index),
            # (r"/func/([0-9]+)", Function),
        ]

        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
        )

        super(Application, self).__init__(handlers, **settings)

    def format_func(self, ftup):
        rank, probname, sysname, method_name = ftup2key(ftup)
        prefix = []
        if rank is not None:
            prefix.append(f"rank {rank}")
        if probname is not None:
            prefix.append(probname)
        path = sysname + '.' if sysname else ''
        prefix.append(path + method_name)
        return shrink(':'.join(prefix))

    def get_function_link(self, ftup):
        fkey = ftup2key(ftup)
        fid, _ = self.func_to_id[fkey]
        return f'<a href="/func/{fid}">{self.format_func(ftup)}</a>'

    def child_iter(self, func_id):
        fkey = self.id_to_func[func_id]
        children = self.self.func_to_id[fkey][1][-1]
        for child, info in sorted(children.items(), key=lambda x: x[0]):
            yield child, info

    def get_index_table_rows(self):
        rows = list(_main_table_row_iter(self.db_fname))
        is_par = False
        for r in rows:
            if r['parallel']:
                is_par = True
                break
        return json.dumps(rows), is_par


class Index(tornado.web.RequestHandler):
    def get(self):
        app = self.application

        table_data, is_par = app.get_index_table_rows()
        is_par = 'true' if is_par else 'false'

        self.write(f'''\
    <html>
    <head>
    </head>
    <link href="/static/tabulator.min.css" rel="stylesheet">
    <script type="text/javascript" src="/static/tabulator.min.js"></script>
    <script type="text/javascript">
    function startup() {{
        let table_data = {table_data};
        let is_par = {is_par};
        let timingheight = (table_data.length > 15) ? 650 : null;

        let timingtable = new Tabulator("#index-timing-table", {{
            // set height of table (in CSS or here), this enables the Virtual DOM and
            // improves render speed dramatically (can be any valid css height value)
            height: timingheight,
            data: table_data, //assign data to table
            layout:"fitDataFill", //"fitColumns", "fitDataFill",
            columns:[ //Define Table Columns
                {{title: "Problem", field:"probname", hozAlign:"left", headerFilter:true,
                  visible:false,}},
                {{title: "System", field:"sysname", hozAlign:"left", headerFilter:true,
                    visible:true,
                    tooltip:function(cell){{
                        return  cell.getRow().getData()["class"];
                    }},
                }},
                {{title: "Method", field:"method", hozAlign:"left", headerFilter:true,
                  visible:true,}},
                {{title: "Class", field:"class", hozAlign:"center", headerFilter:true,
                  visible:false,}},
                {{title: "Rank", field:"rank", hozAlign:"center", headerFilter:true,
                  visible:is_par,}},
                {{title: "Num Procs", field:"nprocs", hozAlign:"center", headerFilter:true,
                  visible:is_par,}},
                {{title: "Tree Level", field:"level", hozAlign:"center", headerFilter:true,
                  visible:true,}},
                {{title: "Parallel Child", field:"parallel", hozAlign:"center", visible: is_par,
                    headerFilter: "tickCross",
                    // need 3 states (checked/unchecked/mixed)
                    headerFilterParams:{{"tristate": true}},
                    headerFilterEmptyCheck: function(value){{return value === null}},
                    formatter:"tickCross",
                    formatterParams: {{
                        crossElement: false,  // gets rid of 'x' elements so only check marks show
                    }},
                }},
                {{title: "Calls", field:"ncalls", hozAlign:"center", headerFilter:false,
                  visible:true,}},
                {{title: "Total Time", field:"total", hozAlign:"right", headerFilter:false,
                  visible:true,}},
                {{title: "Avg Time", field:"avg", hozAlign:"right", headerFilter:false,
                  visible:true,}},
                {{title: "Min Time", field:"tmin", hozAlign:"right", headerFilter:false,
                  visible:true,}},
                {{title: "Max Time", field:"tmax", hozAlign:"right", headerFilter:false,
                  visible:true,}},
                // numcol("Total Time", "total"),
                // numcol("Avg Time", "avg"),
                // numcol("Min Time", "tmin"),
                // numcol("Max Time", "tmax"),
            ]
        }});
    }}
    </script>
    <body onload="startup()">
    <h1>{app.db_fname}</h1>
    <h2>Total time: {format_time(app.total_time)}</h2>
    <div id="index-timing-table"></div>
    </body>
    </html>
    ''')


class Function(tornado.web.RequestHandler):
    def get(self, func_id):
        app = self.application
        func_id = int(func_id)

        def buildFunctionTable(items):
            pass

        self.write(f'''\
    <html>
    <head>
    <style>{app.tabstyle}</style>
    </head>
    <body>
    <a href="/">Home</a>
    <h1>{app.get_function_link(app.id_to_func[func_id])}</h1>
    <h2>Callers</h2>
    <div id="callers_table">
    </div>
    <h2>Callees</h2>
    <div id="callees_table">
    </div>
    </body>
    </html>
    ''')


def view_timing(fname, port=8009):
    """
    Start an interactive web viewer for profiling data.

    Parameters
    ----------
    fname: str
        Name of profile data file.
    port: int
        Port number used by web server.
    """
    app = Application(fname)
    app.listen(port)

    print("starting server on port %d" % port)

    serve_thread  = start_thread(tornado.ioloop.IOLoop.current().start)
    launch_thread = start_thread(lambda: launch_browser(port))

    while serve_thread.isAlive():
        serve_thread.join(timeout=1)


# def view_timing(fname, port=8009):
#     from openmdao.visualization.tables.table_builder import generate_table

#     rows = list(_main_table_row_iter(fname))
#     generate_table(rows, 'tabulator', headers='keys').display()


def _create_timing_file(options):
    timing_managers = timer_mod._timing_managers
    timing_file = options.outfile

    if timing_file is None:
        timing_file = 'timings.pkl'

    nprobs = num_problems()

    timing_data = (timing_managers, timer_mod._total_time, nprobs)

    if MPI is not None:
        # need to consolidate the timing data from different procs
        all_managers = MPI.COMM_WORLD.gather(timing_data, root=0)
        if MPI.COMM_WORLD.rank != 0:
            return
    else:
        all_managers = [timing_data]

    with open(timing_file, 'wb') as f:
        print(f"Saving timing data to '{timing_file}'.")
        pickle.dump(all_managers, f)  # , pickle.HIGHEST_PROTOCOL)


def _timing_setup_parser(parser):
    """
    Allows calling of view_timing from a console script.
    """
    parser.add_argument('file', metavar='file', nargs=1,
                        help='profile file to view.')

    parser.add_argument('-p', '--port', action='store', dest='port',
                        default=8009, type=int,
                        help='port used for web server')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Name of output file where timing data will be stored. By default it '
                        'goes to "timings.pkl".')
    parser.add_argument('--use_context', action='store_true', dest='use_context',
                        help="If set, timing will only be active within a timing_context.")
    parser.add_argument('-v', '--view', action='store', dest='view', default='browser',
                        help="View of the output.  Default view is 'browser', which shows timings "
                        "in the browser. Other options are {_view_options[1:]}.")


def _timing_cmd(options, user_args):
    """
    Implement the 'openmdao timing' command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    filename = _to_filename(options.file[0])
    set_default_prob_name(os.path.basename(filename).rpartition('.')[0])

    if filename.endswith('.py'):
        hooks._register_hook('setup', 'Problem', pre=partial(_set_timer_setup_hook, options))

        with timing_context(not options.use_context):
            _load_and_exec(options.file[0], user_args)

        db_fname = _save_timing_data(options)

    else:  # assume file is a sqlite db
        db_fname = filename
        if options.use_context:
            issue_warning(f"Since given file '{options.file[0]}' is not a python script, the "
                          "'--use_context' option is ignored.")

    view = options.view.lower()
    if view == 'browser':
        view_timing(db_fname, port=options.port)
    # elif view == 'text':
    #     for method in options.funcs:
    #         ret = view_MPI_timing(timing_file, method=method, out_stream=sys.stdout)
    #         if ret is None:
    #             issue_warning(
    #                 f"Could find no children of a ParallelGroup running method '{method.name}'.")
    # elif view == 'dump':
    #     view_timing_dump(timing_file, out_stream=sys.stdout)
    elif view == 'none':
        pass
    else:
        issue_warning(f"Viewing option '{view}' ignored. Valid options are "
                      f"{_view_options}.")
