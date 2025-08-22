from collections import defaultdict
import random
from copy import deepcopy
from sql_util.dbinfo import get_all_db_info_path, get_process_order
from sql_util.run import exec_db_path
from sql_util.writedb import write_db_path

def random_delete(tc2element, order, child2parent, t2c, delete_ratio):
    tc2delete = defaultdict(list)
    for t in order:
        assert len(set([len(tc2element[(t, c)]) for c in t2c[t]])) == 1
        delete_indices = set()
        for c in t2c[t]:
            if (t, c) in child2parent:
                pt, pc = child2parent[(t, c)]
                assert (pt, pc) in tc2delete
                delete_indices.update([i for i, element in enumerate(tc2element[(t, c)])
                                       if element in tc2delete[(pt, pc)]])
                #print((t, c), (pt, pc))
                #print(tc2delete[(pt, pc)])
                #print([(i, element) for i, element in enumerate(tc2element[(t, c)])
                #       if element in tc2delete[(pt, pc)]])
                #print('*' * 80)
        length = len(tc2element[(t, t2c[t][0])])
        delete_num = int(length * delete_ratio)
        if len(delete_indices) < delete_num:
            delete_indices.update(random.sample([i for i in range(length) if i not in delete_indices],
                                                 k=delete_num - len(delete_indices)))
        assert len(delete_indices) >= delete_num
        if len(delete_indices) == length:
            return False
        for c in t2c[t]:
            if (t, c) in child2parent.values():
                tc2delete[(t,c)] = [tc2element[(t, c)][i] for i in range(length) if i in delete_indices]
            tc2element[(t, c)] = [tc2element[(t, c)][i] for i in range(length) if i not in delete_indices]
    return tc2element


def write_db(table_column2elements, db_path, output_path):
    table2column2elements = defaultdict(dict)
    for (t, c), elements in table_column2elements.items():
        table2column2elements[t][c] = elements
    write_db_path(db_path, output_path, table2column2elements, overwrite=True)


import math
def sample_database(db_path, sql_list, db_num, output_id):
    table_column_properties, child2parent, table_column2elements = get_all_db_info_path(db_path)
    table_column_order, table_order = get_process_order(child2parent, table_column_properties)
    #print(child2parent)
    order = []
    for t in table_order:
        order += list(t)
    t2c = defaultdict(list)

    for t, c in table_column2elements.keys():
        t2c[t].append(c)

    distinguish_num = []
    initial_db = None
    for db_id in range(db_num):
        table_column2elements = generate_from_50(table_column2elements, order, child2parent, t2c, db_path, sql_list)
        if table_column2elements is None:
            if initial_db is not None:
                table_column2elements = initial_db
            else:
                return None
        else:
            if initial_db is None:
                initial_db = deepcopy(table_column2elements)

        goal_num = len(sql_list)
        for i in range(19):
            num = 0
            max_num = 0
            max_table_column2elements = None
            #print('i: ', i)
            if max([len(v) for v in table_column2elements.values()]) < 10:
                break
            abandon = False
            while True:
                #print('num: ', num, 'i:', i)
                num += 1
                old_table_column2elements = deepcopy(table_column2elements)
                table_column2elements = random_delete(table_column2elements, order, child2parent, t2c, delete_ratio=0.2)
                #m = defaultdict(list)
                if table_column2elements is False:
                    table_column2elements = old_table_column2elements
                    continue
                output_path = '{}_{}'.format(db_path, 80 - 20 * i)

                write_db(table_column2elements, db_path, output_path)
                result_set = set()
                for sql in sql_list:
                    _, result = exec_db_path(output_path, sql)
                    #m[tuple(result)].append(prob)
                    #sql2result[sql] = tuple(result)
                    result_set.add(tuple(result))
                diff_num = len(result_set)
                if diff_num == goal_num:
                    break
                elif diff_num > max_num:
                    max_num = diff_num
                    max_table_column2elements = deepcopy(table_column2elements)
                if num > 1000:
                    if max_num <= 1:
                        abandon = True
                        break
                    table_column2elements = max_table_column2elements
                    abandon = True
                    goal_num = max_num
                    break
                table_column2elements = old_table_column2elements
            if abandon:
                print(i, 'abandon', len(sql_list), goal_num, max_num)

        distinguish_num.append(goal_num)
        write_db(table_column2elements, db_path, '{}_sample_{}_{}'.format(db_path, db_id, output_id))
    return ['{}_sample_{}_{}'.format(db_path, db_id, output_id) for db_id in range(db_num)], distinguish_num


def generate_from_50(tc2element, order, child2parent, t2c, db_path, sql_list):
    table_size = 50
    parent2child = {parent: child for child, parent in child2parent.items()}
    ret_tc2element = defaultdict(list)
    fail = 0
    while True:
        print(table_size)
        for t in reversed(order):
            assert len(set([len(tc2element[(t, c)]) for c in t2c[t]])) == 1
            index = set()
            for c in t2c[t]:
                if (t, c) in parent2child:
                    childt, childc = parent2child[(t, c)]
                    child_elements = ret_tc2element[(childt, childc)]
                    index.update([i for i, element in enumerate(tc2element[(t, c)]) if element in child_elements])
            if len(index) < table_size:
                index.update(random.sample([i for i in range(len(tc2element[(t, t2c[t][0])])) if i not in index],
                                           k=min(table_size - len(index),
                                           len(tc2element[(t, t2c[t][0])]) - len(index) )))

            for c in t2c[t]:
                ret_tc2element[(t, c)] = [tc2element[(t, c)][i] for i in index]

        output_path = '{}_sample'.format(db_path)
        write_db(ret_tc2element, db_path, '{}_sample'.format(db_path))
        result_set = set()
        for sql in sql_list:
            _, result = exec_db_path(output_path, sql)

            result_set.add(tuple(result))

        diff_num = len(result_set)
        if diff_num == len(sql_list):
            return ret_tc2element
        else:
            fail += 1
            table_size *= 2
            table_size = min(table_size, 400)
        if fail == 20: return None
        '''
        if fail == 5:
            table_size *= 2
            fail = 0
        '''
