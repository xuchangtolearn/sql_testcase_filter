from collections import defaultdict
from sql_util.dbinfo import get_all_db_info_path, get_process_order
import random
import os
from fuzz.sample_small_database import write_db

def generate_database(db_path, output_path, table_size):
    try:
        table_column_properties, child2parent, table_column2elements = get_all_db_info_path(db_path)
    except:
        return False
    table_column_order, table_order = get_process_order(child2parent, table_column_properties)
    order = []
    for t in table_order:
        order += list(t)
    t2c = defaultdict(list)

    for t, c in table_column2elements.keys():
        t2c[t].append(c)

    parent2child = {parent: child for child, parent in child2parent.items()}
    ret_tc2element = defaultdict(list)

    for t in reversed(order):
        assert len(set([len(table_column2elements[(t, c)]) for c in t2c[t]])) == 1
        #if number_range == -1:
        index = set()
        for c in t2c[t]:
            if (t, c) in parent2child:
                childt, childc = parent2child[(t, c)]
                child_elements = ret_tc2element[(childt, childc)]
                index.update([i for i, element in enumerate(table_column2elements[(t, c)]) if element in child_elements])
        
        # !添加动态调整 table_size 的逻辑
        total_rows = len(table_column2elements[(t, t2c[t][0])])
        if total_rows > 5000:
            table_size_current = random.randint(2000, 5000)
        else:
            table_size_current = total_rows
        table_size = table_size_current

        if len(index) < table_size:
            index.update(random.sample([i for i in range(len(table_column2elements[(t, t2c[t][0])])) if i not in index],
                                       k=min(table_size - len(index),
                                             len(table_column2elements[(t, t2c[t][0])]) - len(index) )))

        for c in t2c[t]:
            ret_tc2element[(t, c)] = [table_column2elements[(t, c)][i] for i in index]

    if os.path.exists(output_path):
        os.remove(output_path)
    write_to_db_tc2element = defaultdict(list)

    for t, c in table_column_properties.keys():
        write_to_db_tc2element[(t, c)] = ret_tc2element[(t, c)]

    write_db(write_to_db_tc2element, db_path, output_path)
    return True