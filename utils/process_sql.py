################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json
import sqlite3
from nltk import word_tokenize
import re

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as', 'inner')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'is not null', 'is null')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg', 'date', 'time')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')
SQL_FUNCTIONS = ('strftime')
ignore_name = []



class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        # 打印idMap
        # for key, val in idMap.items():
        #     print(f"{key}: {val}")
        return idMap

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

def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        table_quoted = quote_if_reserved(table)
        cursor.execute("PRAGMA table_info({})".format(table_quoted))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry['table'].lower())
        cols = [str(col['column_name'].lower()) for col in entry['col_data']]
        schema[table] = cols

    return schema


def tokenize(string):
    string = str(string)
    
    # Step 1: 抽取反引号中的字段名 `...`，替换为占位符
    field_vals = {}
    def field_repl(match):
        field = match.group(0)
        key = f"__field_{len(field_vals)}__"
        field_vals[key] = field
        return key

    string = re.sub(r'`[^`]+`', field_repl, string)
    
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # print(toks)
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]
        for key, val in field_vals.items():
            if key in toks[i]:
                toks[i] = toks[i].replace(key, val)

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

    print(toks)
    return toks


def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        alias[toks[idx+1]] = toks[idx-1]
    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
    tok = toks[start_idx]
    tok = tok.lower().strip().strip('`') 

    if tok in ignore_name:
        return start_idx + 1, None
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        col = col.lower().strip().strip('`')
        key = tables_with_alias[alias] + "." + col
        # print(f"key: {key} in {schema.schema[tables_with_alias[alias]]}")
        return start_idx+1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        # for item in schema.schema[table]:
        #     print(f"item: {item} in {schema.schema[table]}")
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx+1, schema.idMap[key]

    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1
        
    if toks[idx] == "cast":
        idx += 1
        assert idx < len_ and toks[idx] == '(', "'(' not found after 'cast'"
        idx += 1
        idx, col_id = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == 'as', "'as' not found"
        idx += 1
        # 解析目标数据类型
        target_data_type = toks[idx]
        idx += 1
        assert idx < len_ and toks[idx] == ')', "')' not found"
        idx += 1
        return idx, ('cast', col_id , isDistinct)
    
    if toks[idx] == "rank":
        if toks[idx+1] == '(' and toks[idx+2] == ')' and toks[idx+3] == 'over':
            idx += 4
            assert toks[idx] == '(', "'(' not found after 'over'"
            idx += 1
            p_col_id = None
            if toks[idx] == 'partition' and toks[idx+1] == 'by':
                idx += 2
                idx, p_col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
            assert toks[idx] == 'order', "'order' not found in RANK() OVER"
            idx += 1
            assert toks[idx] == 'by', "'by' not found after 'order' in RANK() OVER"
            idx += 1
            
            idx, order_col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

            if idx < len_ and toks[idx] in ('desc', 'asc'):
                order_dir = toks[idx]
                idx += 1
            else:
                order_dir = 'asc'  # default

            assert toks[idx] == ')', "')' not found to close OVER clause"
            idx += 1
            if idx < len_ and toks[idx] == 'as':
                ignore_name.append(toks[idx+1])
                idx += 2
            if p_col_id is not None:
                return idx, ('rank', p_col_id, order_col_id, order_dir)
            return idx, ('rank', order_col_id, order_dir)

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True

    if toks[idx] in SQL_FUNCTIONS:
        func_id = SQL_FUNCTIONS.index(toks[idx])
        func_name = toks[idx]
        idx += 1
        assert toks[idx] == '(', f"Expected ( after function {func_name}"
        idx += 1
        param_start = idx
        paren_count = 1  # To count nested parentheses
        while idx < len(toks) and paren_count > 0:
            if toks[idx] == "distinct":
                idx += 1
                isDistinct = True
            if toks[idx] == '(': 
                paren_count += 1
            elif toks[idx] == ')': 
                paren_count -= 1
            idx += 1
        param_start = idx - 2
        _, col_id = parse_col(toks, param_start, tables_with_alias, schema, default_tables)
        # You can handle the parameters of the function if needed
        return idx, (func_id, col_id, isDistinct)
    
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        # 尝试解析为列，否则解析为常量
        try:
            idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        except:
            idx += 1
            col_unit2 = None  # 允许右操作数是常量（如数字或字符串）
            
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val, _ = parse_sql(toks, idx, tables_with_alias, schema)
    elif "\"" in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
        if isBlock:
            while idx < len(toks) and toks[idx] != ')':
                idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
            if isBlock:
                while idx < len(toks) and toks[idx] != ')':
                    idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
                and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and toks[end_idx] not in JOIN_KEYWORDS:
                    end_idx += 1

            idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
            idx = end_idx

    if isBlock:
        #print(idx, toks)
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []

    while idx < len_:
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        not_op = False
        if toks[idx] == 'not':
            not_op = True
            idx += 1

        if toks[idx] == 'is':
            if idx + 2 < len_ and toks[idx+1] == 'not' and toks[idx+2] == 'null':
                op_id = WHERE_OPS.index('is not null')
                idx += 3
                val1 = val2 = None
                conds.append((not_op, op_id, val_unit, val1, val2))
            elif idx + 1 < len_ and toks[idx+1] == 'null':
                op_id = WHERE_OPS.index('is null')
                idx += 2
                val1 = val2 = None
                conds.append((not_op, op_id, val_unit, val1, val2))
                
        else:
            assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
            op_id = WHERE_OPS.index(toks[idx])
            idx += 1
            val1 = val2 = None
            if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
                idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                assert toks[idx] == 'and'
                idx += 1
                idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            else:  # normal case: single value
                idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                val2 = None

            conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    # print(f"tokens:{toks}")
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if_select = False
        if toks[idx] == 'select':
            idx, sql, default_tables = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
            if_select = True
        else:
            if idx < len_ and toks[idx] == 'inner':  # If 'inner' is found, treat as part of 'join'
                idx += 1  # Skip 'inner'
                assert toks[idx] == 'join', "'join' not found after 'inner'"
            if idx < len_ and toks[idx] == 'join':
                idx += 1  # skip join
            if idx+1 < len_ and (toks[idx] == 'left' or toks[idx] == 'right') and toks[idx+1] == 'join':
                idx += 2
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'],table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        
        if if_select:   
            alias = None
            print(f"toks[idx+1]:{toks[idx+1]}")
            if idx < len_ and toks[idx] == 'as':
                alias = toks[idx + 1]
                idx += 2
                print(f"aaa:toks[idx]:{toks[idx]}")
            elif idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
                alias = toks[idx]
                idx += 1
            # if alias:
            #     tables_with_alias[alias] = alias
            #     default_tables.append(alias)
            
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc' # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx-1])

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema):
    isBlock = False # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(toks, idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['orderBy'] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql, _ = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql, default_tables


def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(schema, query):
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    print(f"query: {query}")
    _, sql, _ = parse_sql(toks, 0, tables_with_alias, schema)

    return sql


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx

#import mysqlparse
def post_process(sql, db):
    return sql
    sql = sql.replace('inner join', 'join')
    sql = sql.replace('INNER JOIN', 'JOIN')
    #mysqlparse.parse(sql)

    assert len(db['column_names']) == len(db['column_names_original']) and \
           len(db['table_names']) == len(db['table_names_original'])
    for name, original_name in zip(db['column_names'][1:], db['column_names_original'][1:]):
        name = name[1].replace(' ', '_')
        original_name = original_name[1].lower()
        def replace(match):
            return match.group(1) + original_name + match.group(2)

        sql = re.sub('( |\(|\.|,|^)' + name + '( |\)|\.|,|$)', replace, sql)
    for name, original_name in zip(db['table_names'][1:], db['table_names_original'][1:]):
        name = name.replace(' ', '_').lower()
        original_name = original_name.lower()
        def replace(match):
            return match.group(1) + original_name + match.group(2)
        sql = re.sub('( |\(|\.|^)' + name + '( |\)|\.|$)', replace, sql)

    return sql