from fuzz.base_fuzz import generate_dbfuzzer
from utils.execution_utils import execute_sql_with_time_limit, result_eq, execute_sql_inside_the_process_without_time_limit
from fuzz.db_generate_by_column import generate_database
import os
from sql_util.writedb import write_db_path
from sql_util.dbinfo import get_all_db_info_path
'''
    tables = dbfuzzer.get_fuzz()
    write_db_path(orig_path, target_path, tables, overwrite=True)
    assert os.path.exists(target_path), 'path %s does not exists.' % target_path

'''

def generate_db_to_distinguish_queries(sql, neighbor_queries, orig_db_path, data_index, data_type, db_type, test_case_num,
                                       table_size=5, MAX_NUM=100):
    all_db = []
    try_num = 0
    dir_path = os.path.dirname(orig_db_path)
    # dir_path = os.path.dirname(dir_path)
    test_case_dir = f'{db_type}_test_case'
    if not os.path.exists(os.path.join(dir_path, test_case_dir)):
        os.mkdir(os.path.join(dir_path, test_case_dir))

    # fuzzing和select values的实现是类似的，都是用的fuzzer
    if db_type in ['fuzzing', 'select_values']:
        fuzzer = generate_dbfuzzer(orig_db_path, [sql], table_size,
                                   fuzzing=(db_type == 'fuzzing'))

    # 并不是所有生成的数据库都满足要求，因此最多尝试生成MAX_NUM=100次，如果得不到足够数量的数据库也会停止
    while len(all_db) < test_case_num and try_num < MAX_NUM:
        #print('test_num:', test_num, 'neighbor_queries:', len(neighbor_queries))
        generate_path = os.path.join(dir_path, test_case_dir, 'db_{}_{}_{}.db'.format(data_type, data_index, len(all_db)))
        try_num += 1
        if db_type in ['fuzzing', 'select_values']:
            # 基于fuzzing（随机生成value) 或者 select values （从数据库中选value）合成新数据库
            tables = fuzzer.get_fuzz()
            write_db_path(orig_db_path, generate_path, tables, overwrite=True)
            assert os.path.exists(generate_path), 'path %s does not exists.' % generate_path
        else:
            assert db_type == 'select_columns'
            # 从数据库中以行为单位选择column
            result = generate_database(orig_db_path, generate_path, table_size=table_size)
            assert result, 'path %s does not exists.' % generate_path


        '''
        用于debug，查看合成数据库的具体内容
        _, _, orig_table_column2elements = get_all_db_info_path(orig_db_path)

        _, _, target_table_column2elements = get_all_db_info_path(generate_path)

        for (table_name, column_name), orig_elements in orig_table_column2elements.items():
            target_elements = target_table_column2elements[(table_name, column_name)]

            print(table_name, column_name, target_elements)
        exit(0)
        '''
        # 得到sql在合成数据库上的执行结果
        gt_result = execute_sql_inside_the_process_without_time_limit(sql, generate_path)


        # 最基本的约束：要求执行结果不为空
        if gt_result is None: continue

        # 进一步的约束：要求至少有一个neighbor_sql跟gt_sql执行结果不一样
        flag = False
        for index, neighbor_query in enumerate(neighbor_queries):
            #print(index, delete_list)
            result = execute_sql_inside_the_process_without_time_limit(neighbor_query, generate_path)
            if result is None or not result_eq(gt_result, result):
                flag = True
                break
        if flag:
            all_db.append(generate_path)
        elif try_num == MAX_NUM:
            print(f"[WARNING] {sql} has {len(neighbor_queries)} neighbors, but all of them have the same result with gt_sql")
            all_db.append(generate_path)
    if len(all_db) < test_case_num:
        print(f"[WARNING] Only generated {len(all_db)} of {test_case_num} databases for index {data_index}")
    return all_db



