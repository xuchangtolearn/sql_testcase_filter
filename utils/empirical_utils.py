from sql_util.dbinfo import get_all_db_info_path
from utils.evaluation import get_execution_result
import re
from utils.process_sql import get_schema, Schema, get_sql
import random
from itertools import product
from collections import defaultdict
import os


def get_all_used_tab_col(sql, db_id, db_root = 'data/database', format='filter'):
    def find_select(root):
        ret = set()
        if isinstance(root, dict):
            if 'select' in root:
                assert 'from' in root
                v = root['select']
                for x in v[1]:
                    tab_col_info = re.findall('__(.*?)__', str(x))
                    if 'all' in tab_col_info:
                        v1 = root['from']
                        tab_col_info = re.findall('__(.*?)__', str(v1))
                        if format == 'all' and re.findall("\(0, \(0, \(0, \'__all__\', False\), None\)\)",
                                      str(x)) != []:
                            tab_list = [(info, 'all') for info in tab_col_info if '.' not in info and info != 'all']
                        else:
                            tab_list = [info for info in tab_col_info if '.' not in info and info != 'all']
                        ret.update(tab_list)
            for k, v in root.items():
                ret.update(find_select(v))
        elif isinstance(root, list) or isinstance(root, tuple):
            for x in root:
                ret.update(find_select(x))
        return ret

    # db_path = 'data/database/{}/{}.sqlite'.format(db_id, db_id)
    db_path = os.path.join(db_root, '{}/{}.sqlite'.format(db_id, db_id))
    schema = Schema(get_schema(db_path))
    print(get_schema(db_path))
    ret = set()
    table_column2properties = get_all_db_info_path(db_path)
    print(table_column2properties)
    exit(0)

    sql = get_sql(schema, sql)
    sql_str = str(sql)
    #print(sql)
    tab_set = find_select(sql)
    tab_col_info = re.findall('__(.*?)__', sql_str)
    tab_col_list = [tuple(info.split('.')) for info in tab_col_info if '.' in info]

    assert all([len(x) == 2 for x in tab_col_list])
    ret.update(tab_col_list)
    used_table = set(x[0] for x in ret)
    for k, v in table_column2properties.items():
        if (k[0] in used_table or k[0] in tab_set) and v['PK'] > 0:
            ret.add((k[0].lower(), k[1].lower()))
    
    for tab in tab_set:
        if isinstance(tab, tuple) and tab[1] == 'all':
            for k, v in table_column2properties.items():
                if k[0] == tab[0]:
                    ret.add((k[0].lower(), k[1].lower()))
        elif tab not in [t for t, c in ret]:
            for k, v in table_column2properties.items():
                if k[0] == tab:
                    ret.add((k[0].lower(), k[1].lower()))
                    break
            assert tab in [t for t, c in ret]

    assert len(ret) > 0, (sql, db_id, ret)
    return list(ret)


def get_all_used_tab_col_string_based(sql: str, db_path: str, db_id: str, db_root: str = 'data/database', format: str = 'filter'):
    """
    基于字符串匹配方式提取 SQL 中实际使用到的表-列对
    兼容 get_all_used_tab_col 的接口与返回格式

    :param sql: 原始 SQL 字符串
    :param db_id: 数据库 ID（文件名）
    :param db_root: 数据库根目录
    :param format: 可选，用于兼容原始逻辑（保留但未使用）
    :return: list of (table_name, column_name) 对，列名全部小写
    """
    # db_path = os.path.join(db_root, f"{db_id}/{db_id}.sqlite")
    schema = Schema(get_schema(db_path))
    
    used_tab_col = set()
    sql = sql.lower()

    # 遍历所有表和列
    for table, columns in schema.schema.items():
        table = table.lower()
        if table in sql and '*' in sql:
            used_tab_col.add((table, 'all'))  # 使用了整表
            continue
        for col in columns:
            if col.lower() in sql:
                used_tab_col.add((table, col.lower()))

    # 如果使用了表但没有使用其主键或任何列，可以考虑添加主键
    # （略去主键信息的获取，可根据需要从 schema 外部传入）

    assert len(used_tab_col) > 0, (sql, db_id, used_tab_col)
    return list(used_tab_col)


def predict_process(s):
    ret = []
    s = s.strip(' ').strip('\n').split('\n')
    s = list(filter(lambda x: x.strip(' ').strip('\n') != '', s))

    for x in s:
        try:
            # 先处理非法 markdown 包裹（```）
            if x.strip().startswith('```'):
                x = x.strip('`')
            tmp = eval(x)
        except Exception:
            # 如果 eval 出错，返回目前已有的去重结果
            tmp_set = set()
            safe_ret = []
            for item in ret:
                try:
                    tmp_set.add(item)
                except TypeError:
                    # 不能直接 set 的，手动去重
                    if item not in safe_ret:
                        safe_ret.append(item)
            return list(tmp_set) if len(tmp_set) > len(safe_ret) else safe_ret

        # 统一处理成 tuple
        if isinstance(tmp, list):
            tmp = tuple(tmp)
        elif not isinstance(tmp, tuple):
            tmp = (tmp,)

        if tmp in [(None,), None, [], 0, (0,)]:
            tmp = tuple()

        final = []
        for s in tmp:
            if isinstance(s, str):
                try:
                    t = eval(s)
                    if isinstance(t, int) or isinstance(t, float):
                        final.append(round(t) if isinstance(t, float) else t)
                    else:
                        final.append(s)
                except:
                    final.append(s)
            elif isinstance(s, float):
                final.append(round(s))
            else:
                final.append(s)

        processed = []
        for x in final:
            if isinstance(x, list):
                processed.append(tuple(x))
            else:
                processed.append(x)

        ret.append(tuple(processed))

    # 如果什么都没有，默认加一个空tuple
    if len(ret) == 0:
        ret = [tuple()]

    # 安全去重（防止dict无法set）
    tmp_set = set()
    safe_ret = []
    for item in ret:
        try:
            tmp_set.add(item)
        except TypeError:
            if item not in safe_ret:
                safe_ret.append(item)

    dedup_ret = list(tmp_set) if len(tmp_set) > len(safe_ret) else safe_ret

    # 最后再做一次内部元素转换：如果可以eval为数值就eval
    final_ret = []
    for x in dedup_ret:
        t = []
        for y in x:
            try:
                v = eval(str(y))
                if isinstance(v, int) or isinstance(v, float):
                    t.append(v)
                else:
                    t.append(y)
            except:
                t.append(y)
        final_ret.append(tuple(t))

    return final_ret


def unorder_row(row):
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))

def quick_rej(result1, result2, order_matters):
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)

def get_constraint_permutation(tab1_sets_by_columns, result2):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)

def permute_tuple(element, perm):
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])

def test_suite_judge_result_predict(result1, result2, order_matters):
    if isinstance(result1, str):
        result1 = predict_process(result1)
    if isinstance(result2, str):
        result2 = predict_process(result2)
    #print('!!!', result1, result2)
    if result1 == [()]:
        result1 = []
    if result2 == [()]:
        result2 = []
    if len(result1) == 0 and len(result2) == 0:
        return True


    # if length is not the same, then they are definitely different bag of rows
    if len(result1) != len(result2):
        if len(result1) == 1 and len(result1[0]) == len(result2):
            result1 = [(x,) for x in result1[0]]
        elif len(result2) == 1 and len(result2[0]) == len(result1):
            result2 = [(x, ) for x in result2[0]]
        else:
            return False

    num_cols = len(result1[0])

    # if the results do not have the same number of columns, they are different
    if len(result2[0]) != num_cols:
        return False

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if not quick_rej(result1, result2, order_matters):
        return False

    # the rest of the problem is in fact more complicated than one might think
    # we want to find a permutation of column order and a permutation of row order,
    # s.t. result_1 is the same as result_2
    # we return true if we can find such column & row permutations
    # and false if we cannot
    # tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]
    num_cols = max(
        max((len(row) for row in result1), default=0),
        max((len(row) for row in result2), default=0)
    )

    # 如果列数为 0，不可比，直接返回 False
    if num_cols == 0:
        return False

    # 构造每列对应的值集合，同时过滤掉长度不足的 row
    tab1_sets_by_columns = [
        {row[i] for row in result1 if len(row) > i}
        for i in range(num_cols)
    ]

    # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
    # we decrease the size of the column permutation space by the function get_constraint_permutation
    # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            # in fact the first condition must hold if the second condition holds
            # but the first is way more efficient implementation-wise
            # and we use it to quickly reject impossible candidates
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False

def multiset_eq(l1, l2) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True

def my_get_sql_execution_result(sql, db_path, format='col_order'):
    assert format in ['col_order', 'row_order']
    assert format == 'row_order'
    result = get_execution_result(sql, db_path=db_path, format=format)
    if isinstance(result, bool):
        return False

    if format == 'col_order':
        output = []
        for k, v in result.items():
            output.append((k[1].strip('_'), v))
        o = ''
        for col, v in output:
            o += '({}, {})\n'.format(col, str(v))
        o = o[:-1]
    else:
        o = ''
        for row in result:
            o += '{}\n'.format(row)
        o = o[:-1]
    return o, result


def is_type_number(t):
    t = t.lower()
    return 'float' in t or t == 'integer' or t == 'int' or 'decimal' in t or 'real' in t or 'number' in t