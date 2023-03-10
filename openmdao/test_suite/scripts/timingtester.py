import sqlite3
import sys


from openmdao.visualization.timing_viewer.timer import func_info_iter, func_tree, obj_tree, db2table


# JOIN:
# SELECT EMP_ID, NAME, DEPT FROM COMPANY INNER JOIN DEPARTMENT
#    ON COMPANY.ID = DEPARTMENT.EMP_ID

def main():
    dbname = sys.argv[1]

    with sqlite3.connect(dbname) as con:
        cur = con.cursor()
        cur2 = con.cursor()

        # rank, prob_name, class_name, sys_name, method, level, parallel, nprocs, ncalls, ftime, tmin, tmax
        db2table(sorted(cur.execute("SELECT sys_name, method, ncalls, ftime, tmin, tmax from func_index"), key=lambda x: x[3], reverse=True),
                 format='tabulator', headers=['System', 'Method', 'Calls', 'Total Time', 'Min Time', 'Max Time'])

        # for row in cur.execute("SELECT * from func_index ORDER BY ftime DESC"):
        #     print(row)
        #     for child in cur2.execute(f"SELECT child_name, ncalls, ftime from call_tree WHERE parent_id = {row[0]}"):
        #         print(f"   calls {child}")

        # print('-' * 80)

        # for row in cur.execute("SELECT * from call_tree ORDER BY ftime"):
        #     print(row)

        # print('-' * 80)

        # for child in cur2.execute(f"SELECT DISTINCT child_name, child_id from call_tree"):
        #     print(child[0])
        #     for parent, in cur.execute(f"SELECT parent_name from call_tree WHERE child_id={child[1]}"):
        #         print(f"   called by {parent}")

        # print('-' * 80)

        # id, rank, prob_name, class_name, sys_name, method, level, parallel, nprocs, ncalls, ftime, tmin, tmax
        # for row in cur.execute("SELECT * from func_index ORDER BY level"):
        #     fid = row[0]
        #     sname = row[4]
        #     pname = row[2]
        #     fname = row[5]
        #     rank = row[1]

        #     print(row)
        #     break

        # OD1.fc.conv.fs.totals.base_thermo.props.tp2props.compute_partials
        # func_tree(con, sname, fname, pname, rank)

        # obj_tree(con)

if __name__ == '__main__':
    main()
