import os.path
import pickle as pkl
from typing import Set, List, Tuple, Dict, Any, TypeVar
from collections import OrderedDict
from sql_util.run import exec_db_path_
from sql_util.dbpath import get_value_path
from collections import defaultdict
import random
import re


table_name_query = "SELECT name FROM sqlite_master WHERE type='table';"
column_type_query = "pragma table_info('%s');"
foreign_key_query = "pragma foreign_key_list('%s')"
table_schema_query = "select sql from sqlite_master where type='table' and name='%s'"
select_all_query = "SELECT * from %s;"


def get_values(db_name: str) -> Set[str]:
    values = pkl.load(open(get_value_path(db_name), 'rb'))
    return values


def get_schema_path(sqlite_path: str, table_name: str) -> str:
    _, schema = exec_db_path_(sqlite_path, table_schema_query % table_name)
    schema = schema[0][0]
    return schema


def get_unique_keys(schema: str) -> Set[str]:
    schema_by_list = schema.split('\n')
    unique_keys = set()
    for r in schema_by_list:
        if 'unique' in r.lower():
            unique_keys.add(r.strip().split()[0].upper().replace("\"", '').replace('`', ''))
    return unique_keys


def get_checked_keys(schema: str) -> Set[str]:
    schema_by_list = schema.split('\n')
    checked_keys = set()
    for r in schema_by_list:
        if 'check (' in r or 'check(' in r:
            checked_keys.add(r.strip().split()[0].upper().replace("\"", '').replace('`', ''))
    return checked_keys


def get_table_names_path(sqlite_path: str) -> List[str]:
    table_names = [x[0] for x in exec_db_path_(sqlite_path, table_name_query)[1]]
    return table_names


def extract_table_column_properties_path(sqlite_path: str) \
        -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Tuple[str, str]]]:
    table_names = get_table_names_path(sqlite_path)
    table_name2column_properties = OrderedDict()
    child2parent = OrderedDict()
    for table_name in table_names:
        schema = get_schema_path(sqlite_path, table_name)
        unique_keys, checked_keys = get_unique_keys(schema), get_checked_keys(schema)
        table_name = table_name.lower()
        column_properties = OrderedDict()
        result_type, result = exec_db_path_(sqlite_path, column_type_query % table_name)
        for (
                columnID, column_name, columnType,
                columnNotNull, columnDefault, columnPK,
        ) in result:
            column_name = column_name.upper()
            column_properties[column_name] = {
                'ID': columnID,
                'name': column_name,
                'type': columnType,
                'notnull': columnNotNull,
                'default': columnDefault,
                'PK': columnPK,
                'unique': column_name in unique_keys,
                'checked': column_name in checked_keys
            }
        table_name2column_properties[table_name.lower()] = column_properties

        # extract foreign keys and population child2parent
        result_type, result = exec_db_path_(sqlite_path, foreign_key_query % table_name)
        for (
                keyid, column_seq_id, other_tab_name, this_column_name, other_column_name,
                on_update, on_delete, match
        ) in result:
            # these lines handle a foreign key exception in the test set
            # due to implicit reference
            if other_column_name is None:
                other_column_name = this_column_name

            table_name, other_tab_name = table_name.lower(), other_tab_name.lower()
            this_column_name, other_column_name = this_column_name.upper(), other_column_name.upper()

            # these lines handle a foreign key exception in the test set
            # due to typo in the column name
            if other_tab_name == 'author' and other_column_name == 'IDAUTHORA':
                other_column_name = 'IDAUTHOR'
            if other_tab_name == 'country' and other_column_name == 'COUNTRY_ID':
                other_column_name = 'ID'
            if other_tab_name == 'league' and other_column_name == 'LEAGUE_ID':
                other_column_name = 'ID'

            child2parent[(table_name, this_column_name)] = (other_tab_name, other_column_name)

    # make sure that every table, column in the dependency are in the table.
    dep_table_columns = set(child2parent.keys()) | set(child2parent.values())
    for table_name, column_name in dep_table_columns:
        assert table_name.lower() == table_name, "table name should be lower case"
        assert column_name.upper() == column_name, "column name should be upper case"
        assert table_name in table_name2column_properties, "table name %s missing." % table_name
        assert column_name in table_name2column_properties[table_name], \
            "column name %s should be present in table %s" % (column_name, table_name)

    return table_name2column_properties, child2parent


T = TypeVar('T')
# collapse a two level dictionary into a single level dictionary
def collapse_key(d: Dict[str, Dict[str, T]]) -> Dict[Tuple[str, str], T]:
    result = OrderedDict()
    for k1, v1 in d.items():
        for k2, v2 in v1.items():
            result[(k1, k2)] = v2
    return result


E = TypeVar('E')
def process_order_helper(dep: Dict[E, Set[E]], all: Set[E]) -> List[Set[E]]:
    dep_ks = set(dep.keys())
    for k in dep.values():
        dep_ks |= set(k)
    # assert that all the elements in the dependency relations are in the universe set
    assert len(dep_ks - all) == 0, dep_ks - all
    order = list(my_top_sort({k: v for k, v in dep.items()}))
    if len(order) == 0:
        order.append(set())
    for k in all:
        if k not in dep_ks:
            order[0].add(k)
    s = set()
    for o in order:
        s |= set(o)
    assert len(s) == len(all), (s - all, all - s)
    return order


def my_top_sort(dep: Dict[E, Set[E]]) -> List[Set[E]]:
    order = []
    elements_left = set()
    for child, parents in dep.items():
        elements_left.add(child)
        elements_left |= parents

    while len(elements_left) != 0:
        level_set = set()
        for e in elements_left:
            if e not in dep.keys():
                level_set.add(e)
            else:
                if all(parent not in elements_left for parent in dep[e]):
                    level_set.add(e)
        for e in level_set:
            elements_left.remove(e)
        order.append(level_set)
    return order


# order the columns/tables by foreign key references
def get_process_order(child2parent: Dict[Tuple[str, str], Tuple[str, str]],
                      table_column_properties: Dict[Tuple[str, str], Dict[str, Any]])\
        -> Tuple[List[Set[Tuple[str, str]]], List[Set[str]]]:
    all_table_column = set(table_column_properties.keys())
    dep_child2parent = {c: {p} for c, p in child2parent.items()}
    table_column_order = process_order_helper(dep_child2parent, all_table_column)
    all_table = set([k[0] for k in all_table_column])
    table_child2parent = defaultdict(set)
    for k1, k2 in child2parent.items():
        if k1[0] != k2[0]:
            table_child2parent[k1[0]].add(k2[0])
    table_order = process_order_helper(table_child2parent, all_table)
    return table_column_order, table_order


# load information from the database
# including:
# 1. column_properties: (table_name, column_name) -> column properties
#   where column properties are a map from property_name (str) -> value
# 2. foreign key relations: (table_name, column_name) -> (table_name, column_name)
# 3. column_content: (table_name, column_name) -> list, list of element types.
SQLITE_RESERVED_KEYWORDS = {
    "abort", "action", "add", "after", "all", "alter", "analyze", "and", "as", "asc", "attach", "autoincrement",
    "before", "begin", "between", "by", "cascade", "case", "cast", "check", "collate", "column", "commit", "conflict",
    "constraint", "create", "cross", "current_date", "current_time", "current_timestamp", "database", "default",
    "deferrable", "deferred", "delete", "desc", "detach", "distinct", "drop", "each", "else", "end", "escape", "except",
    "exclusive", "exists", "explain", "fail", "for", "foreign", "from", "full", "glob", "group", "having", "if",
    "ignore", "immediate", "in", "index", "indexed", "initially", "inner", "insert", "instead", "intersect", "into",
    "is", "isnull", "join", "key", "left", "like", "limit", "match", "natural", "no", "not", "notnull", "null", "of",
    "offset", "on", "or", "order", "outer", "plan", "pragma", "primary", "query", "raise", "recursive", "references",
    "regexp", "reindex", "release", "rename", "replace", "restrict", "right", "rollback", "row", "savepoint", "select",
    "set", "table", "temp", "temporary", "then", "to", "transaction", "trigger", "union", "unique", "update", "using",
    "vacuum", "values", "view", "virtual", "when", "where", "with", "without"
}

def quote_if_reserved(name: str) -> str:
    if name.lower() in SQLITE_RESERVED_KEYWORDS:
        return f'"{name}"'
    return name

def get_all_db_info_path(sqlite_path: str) \
        -> Tuple[
            Dict[Tuple[str, str], Dict[str, Any]],
            Dict[Tuple[str, str], Tuple[str, str]],
            Dict[Tuple[str, str], List],
        ]:
    table_name2column_properties, child2parent = extract_table_column_properties_path(sqlite_path)

    table_name2content = OrderedDict()
    for table_name in table_name2column_properties:
        result_type, result = exec_db_path_(sqlite_path, select_all_query % table_name)
        # ensure that table retrieval succeeds
        if result_type == 'exception':
            raise result
        table_name2content[table_name] = result

    table_name2column_name2elements = OrderedDict()
    for table_name in table_name2column_properties:
        column_properties, content = table_name2column_properties[table_name], table_name2content[table_name]
        # initialize the map from column name to list of elements
        table_name2column_name2elements[table_name] = OrderedDict((column_name, []) for column_name in column_properties)
        # ensure that the number of columns per row
        # is the number of columns
        if len(content) > 0:
            assert len(content[0]) == len(column_properties)
        for row in content:
            for column_name, element in zip(column_properties, row):
                table_name2column_name2elements[table_name][column_name].append(element)

    return collapse_key(table_name2column_properties), child2parent, collapse_key(table_name2column_name2elements)

def get_table_size(table_column_elements: Dict[Tuple[str, str], List]) -> Dict[str, int]:
    table_name2size = OrderedDict()
    for k, elements in table_column_elements.items():
        table_name = k[0]
        if table_name not in table_name2size:
            table_name2size[table_name] = len(elements)
    return table_name2size


def get_primary_keys(table_column_properties: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, List[str]]:
    table_name2primary_keys = OrderedDict()
    for (table_name, column_name), property in table_column_properties.items():
        if table_name not in table_name2primary_keys:
            table_name2primary_keys[table_name] = []
        if property['PK'] != 0:
            table_name2primary_keys[table_name].append(column_name)
    return table_name2primary_keys


def get_indexing_from_db(db_path: str, shuffle=True) -> Dict[str, List[Dict[str, Any]]]:
    table_column_properties, _, _ = get_all_db_info_path(db_path)
    all_tables_names = {t_c[0] for t_c in table_column_properties}

    table_name2indexes = {}
    for table_name in all_tables_names:
        column_names = [t_c[1] for t_c in table_column_properties if t_c[0] == table_name]
        selection_query = 'select ' + ', '.join(['"%s"' % c for c in column_names]) + ' from "' + table_name + '";'
        retrieved_results = exec_db_path_(db_path, selection_query)[1]
        table_name2indexes[table_name] = [{name: e for name, e in zip(column_names, row)} for row in retrieved_results]
        if shuffle:
            random.shuffle(table_name2indexes[table_name])
    return table_name2indexes


def print_table(table_name, column_names, rows):
    print('table:', table_name)
    num_cols = len(column_names)
    template = " ".join(['{:20}'] * num_cols)
    print(template.format(*column_names))
    for row in rows:
        print(template.format(*[str(r) for r in row]))


def database_pprint(path):
    tc2_, _, _ = get_all_db_info_path(path)
    table_column_names = [tc for tc in tc2_.keys()]
    table_names = {t_c[0] for t_c in table_column_names}
    for table_name in table_names:
        column_names = [c for t, c in table_column_names if t == table_name]
        elements_by_column = []
        for column_name in column_names:
            _, elements = exec_db_path_(path, 'select {column_name} from {table_name}'.format(column_name=column_name, table_name=table_name))
            elements_by_column.append([e[0] for e in elements])
        rows = [row for row in zip(*elements_by_column)]
        print_table(table_name, column_names, rows)



def get_total_size_from_indexes(table_name2indexes: Dict[str, List[Dict[str, Any]]]) -> int:
    return sum([len(v) for _, v in table_name2indexes.items()])


def get_total_size_from_path(path):
    _, _, table_column2elements = get_all_db_info_path(path)
    return sum([v for _, v in get_table_size(table_column2elements).items()])


def get_db_info_prompt(db_id, table_col_list=[], format='csv'):
    #print(db_id)
    assert format in ['csv', 'insert']
    if os.path.exists(db_id):
        table_column2properties, column_references, table_column2elements = get_all_db_info_path(db_id)
    else:
        table_column2properties, column_references, table_column2elements = get_all_db_info_path('data/database/{}/{}.sqlite'.format(db_id, db_id))

    table2elements = defaultdict(list)
    for k, v in table_column2elements.items():
        table, column = k
        table2elements[table].append((column, v))
    info = ''
    if format == 'csv':
        for k, v in table2elements.items():
            if table_col_list != []:
                if k not in [tab.lower() for tab, col in table_col_list]:
                    continue
            info += 'Table: {}\n'.format(k)
            col_list = []
            if table_col_list != []:
                for tab, col in table_col_list:
                    if tab.lower() == k.lower() and (col.upper() in [c for c, _ in v]):
                        col_list.append(col.upper())
                    if col == 'all':
                        col_list = [c for c, _ in v]
            else:
                col_list = [c for c, _ in v]
            info += 'Columns: {}\n'.format([c.lower() for c, _ in v if c in col_list])
            info += 'Number of rows: {}\n'.format(len(v[0][1]))
            info += 'Rows:\n'
            for i in range(len(v[0][1])):
                info += '{}\n'.format([e[i] for c, e in v if c in col_list])
            info += '\n'
        return info[:-1]
    else:
        for k, v in table2elements.items():
            if table_col_list != []:
                if k not in [tab for tab, col in table_col_list]:
                    continue
            table_info = '''CREATE TABLE {} (\n{});\n'''
            col_list = []
            if table_col_list != []:
                for tab, col in table_col_list:
                    if tab == k:
                        col_list.append(col.upper())
                        assert col.upper() in [c for c, _ in v]
            else:
                col_list = [c for c, _ in v]
            column_info = ''
            pk_col = []
            insert_info = ''

            len_list = [len(e) for c, e in v]
            assert len(set(len_list)) == 1
            row_num = len_list[0]
            value_list = [[] for _ in range(row_num)]
            for c, e in v:
                if c not in col_list: continue
                property = table_column2properties[(k, c)]
                col_type = property['type']
                if property['PK'] > 0:
                    pk_col.append(c.lower())
                column_info += '{} {},\n'.format(c.lower(), col_type.lower())

                for tmp_index in range(row_num):
                    value_list[tmp_index].append(e[tmp_index])

            if pk_col != '':
                column_info += 'primary key({}),\n'.format(', '.join(pk_col))

            for x, y in column_references.items():
                t1, c1 = x[0].lower(), x[1].lower()
                t2, c2 = y[0].lower(), y[1].lower()
                if (table_col_list != [] and (t1, c1) in table_col_list and (t2, c2) in table_col_list \
                    and t1 == k.lower()) or (table_col_list == [] and t1 == k.lower()):
                    column_info += \
                        'foreign key({}) references {}({}),\n'.format(c1, t2, c2)


            if column_info.endswith(',\n'):
                column_info = column_info[:-2] + '\n'
            table_info = table_info.format(k, column_info)
            #print(table_info)

            for row in value_list:
                tmp = ''
                row = [x if x is not None else 'NULL' for x in row]
                if len(row) == 0:
                    tmp = '()'
                elif len(row) == 1:
                    tmp = '({})'.format(repr(row[0]))
                else:
                    tmp = str(tuple(row))
                tmp = re.sub("\'NULL\'", "NULL", tmp)
                tmp = re.sub('\"NULL\"', "NULL", tmp)
                insert_info += "insert into {} values {};\n".format(k, tmp)
            #print(insert_info)



            info += table_info + insert_info + '\n'
        #print('*' * 80)
        return info[:-1]