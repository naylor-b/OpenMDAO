
import os
import time
from functools import partial
import webbrowser
import threading
import json
import sqlite3

import tornado.ioloop
import tornado.web

import openmdao.utils.hooks as hooks
from openmdao.core.problem import _problem_names, set_default_prob_name, num_problems
from openmdao.visualization.timing_viewer.timer import timing_context, _set_timer_setup_hook, \
    _save_timing_data, _main_table_row_iter, _global_info, id2func_info, children_iter
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


# def ftup2key(ftup):
#     # (rank, probname, sysname, method_name)
#     return (ftup[0], ftup[1], ftup[3], ftup[7])


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
        self.is_par = None
        self.index_rows = None

        self.total_time, self.nprocs = _global_info(db_fname)

        handlers = [
            (r"/", Index),
            (r"/function/([0-9]+)", Function),
        ]

        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
        )

        self.common_js = """
        function val_cell_formatter(cell, formatterParams, onRendered) {
            let val = cell.getValue();
            if (val === "") {
                return "";
            }
            return String(val.toFixed(9));
            //return vsprintf('%g8.3', [val]);
        }

        function val_sorter(a, b, aRow, bRow, column, dir, sorterParams) {
            if (a === "") {
                a = -1e99;
            }
            if (b === "") {
                b = -1e99;
            }
            return a - b;
        }

        function numcol(title, field, sort) {
            let hsort = false;
            if (sort) {
                sort = val_sorter;
                hsort = true;
            }
            return {
                title: title,
                field: field,
                hozAlign: "right",
                visible: true,
                headerFilter: false,
                formatter: val_cell_formatter,
                headerSort: hsort,
                sorter: sort,
            }
        }

        function make_table(tdata, colnames, is_par, theight, tid, idname, tlayout="fitDataFill") {
          let filt = true;
          let sort = true;
          let hsort = true;

          if (tdata.length < 2) {
            filt = false;
            sort = false;
            hsort = false;
          }

          let tdict = {
            probname: {title: "Problem", field:"probname", hozAlign:"left", headerFilter:filt,
                       visible:false, headerSort:hsort},
            sysname: {title: "System", field:"sysname", hozAlign:"left", headerFilter:filt,
                      visible:true, headerSort:hsort,
                      tooltip:function(cell){
                        return cell.getRow().getData()["class"];
                      },
            },
            method: {title: "Method", field:"method", hozAlign:"left", headerFilter:filt,
                     visible:true, headerSort:hsort,
            },
            class: {title: "Class", field:"class", hozAlign:"center", headerFilter:filt,
                    visible:false, headerSort:hsort,},
            rank: {title: "Rank", field:"rank", hozAlign:"center", headerFilter:filt,
                   visible:is_par, headerSort:hsort,},
            nprocs: {title: "Num Procs", field:"nprocs", hozAlign:"center", headerFilter:filt,
                     visible:is_par, headerSort:hsort,},
            level: {title: "Tree Level", field:"level", hozAlign:"center", headerFilter:filt,
                    visible:true, headerSort:hsort,},
            parallel:  {title: "Parallel Child", field:"parallel", hozAlign:"center",
                        visible: is_par, headerSort:hsort,
                        headerFilter: "tickCross",
                        // need 3 states (checked/unchecked/mixed)
                        headerFilterParams:{"tristate": true},
                        headerFilterEmptyCheck: function(value){return value === null},
                        formatter:"tickCross",
                        formatterParams: {
                          crossElement: false, // get rid of 'x' elements so only check marks show
                        },
            },
            ncalls: {title: "Calls", field:"ncalls", hozAlign:"center", headerFilter:false,
                     visible:true, headerSort:hsort,},
            total: numcol("Total Time", "total", sort),
            avg: numcol("Avg Time", "avg", sort),
            tmin: numcol("Min Time", "tmin", sort),
            tmax: numcol("Max Time", "tmax", sort),
          }

        if (idname !== null) {
            tdict["method"]["formatter"] = "link";
            tdict["method"]["formatterParams"] = {
                labelField: "method",
                url: function(cell) {
                    return "/function/" + String(cell.getRow().getData()[idname]);
                },
            }
        }

        let tcols = colnames.map(function (colname) {
            return tdict[colname];
        });

        let timingtable = new Tabulator(tid, {
            // set height of table (in CSS or here), this enables the Virtual DOM and
            // improves render speed dramatically (can be any valid css height value)
            height: theight,
            data: tdata, //assign data to table
            layout: tlayout,  // "fitDataFill", "fitColumns", "fitDataFill",
            columns: tcols,
        });

        return timingtable;
        }

    """

        super(Application, self).__init__(handlers, **settings)

    # def format_func(self, ftup):
    #     rank, probname, sysname, method_name = ftup2key(ftup)
    #     prefix = []
    #     if rank is not None:
    #         prefix.append(f"rank {rank}")
    #     if probname is not None:
    #         prefix.append(probname)
    #     path = sysname + '.' if sysname else ''
    #     prefix.append(path + method_name)
    #     return shrink(':'.join(prefix))

    # def get_function_link(self, ftup):
    #     fkey = ftup2key(ftup)
    #     fid, _ = self.func_to_id[fkey]
    #     return f'<a href="/func/{fid}">{self.format_func(ftup)}</a>'

    def child_iter(self, func_id):
        fkey = self.id_to_func[func_id]
        children = self.self.func_to_id[fkey][1][-1]
        for child, info in sorted(children.items(), key=lambda x: x[0]):
            yield child, info

    def get_index_table_rows(self):
        if self.index_rows is None:
            self.index_rows = rows = list(_main_table_row_iter(self.db_fname))
            if self.is_par is None:
                self.is_par = False
                for r in rows:
                    if r['parallel']:
                        self.is_par = True
                        break

        return json.dumps(self.index_rows)


class Index(tornado.web.RequestHandler):
    def get(self):
        app = self.application

        table_data = app.get_index_table_rows()
        is_par = 'true' if app.is_par else 'false'

        self.write(f'''\
    <html>
    <head>
    </head>
    <link href="/static/tabulator.min.css" rel="stylesheet">
    <script type="text/javascript" src="/static/tabulator.min.js"></script>
    //<script type="text/javascript" src="/static/sprintf.min.js"></script>
    <script type="text/javascript">
    function startup() {{
        let table_data = {table_data};
        let is_par = {is_par};
        let timingheight = (table_data.length > 15) ? 650 : null;

        {app.common_js}

        let colnames = ["probname", "sysname", "method", "class", "rank", "nprocs", "level",
                        "parallel", "ncalls", "total", "avg", "tmin", "tmax"]
        let timingtable = make_table(table_data, colnames, is_par, timingheight,
                                     "#index-timing-table", "id");

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
        is_par = 'true' if app.is_par else 'false'
        func_id = int(func_id)

        parent_row = app.index_rows[func_id - 1].copy()
        parent_row['id'] = 0
        parent_rows = [parent_row]

        with sqlite3.connect(app.db_fname) as dbcon:

            child_rows = []
            for i, tup in enumerate(children_iter(dbcon, func_id)):
                child_id, _, ncalls, ftime = tup
                child_row = app.index_rows[int(child_id) - 1].copy()
                child_row['id'] = i
                child_row['child_id'] = child_id
                child_row['ncalls'] = ncalls
                child_row['total'] = ftime
                child_rows.append(child_row)


        self.write(f'''\
    <html>
    <head>
    </head>
    <link href="/static/tabulator.min.css" rel="stylesheet">
    <script type="text/javascript" src="/static/tabulator.min.js"></script>
    <script type="text/javascript">
    function startup() {{
        let parent_data = {parent_rows};
        let table_data = {child_rows};
        let is_par = {is_par};
        let timingheight = (table_data.length > 15) ? 650 : null;

        {app.common_js}

        let colnames = ["probname", "sysname", "method", "class", "rank", "nprocs", "level",
                        "parallel", "ncalls", "total", "avg", "tmin", "tmax"]
        let parenttable = make_table(parent_data, colnames, is_par, null, "#func_table", null);

        colnames = ["probname", "sysname", "method", "class", "rank", "nprocs", "level",
                    "parallel", "ncalls", "total"]
        let childtable = make_table(table_data, colnames, is_par, timingheight,
                                    "#callees_table", "child_id");

    }}
    </script>
    <body onload="startup()">
        <a href="/">Home</a>
        <h2>Function</h2>
        <div id="func_table"></div>
        <h2>Callers</h2>
        <div id="callers_table"></div>
        <h2>Callees</h2>
        <div id="callees_table"></div>
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
