import random
import multiprocessing
from collections import OrderedDict
from typing import Dict, List, Tuple, Set, TypeVar
from fuzz.date import DateFuzzer
from fuzz.number import NumberFuzzer, isint
from fuzz.bool import BoolFuzzer, BitFuzzer
from fuzz.time import TimeFuzzer
from fuzz.base import BaseFuzzer
from fuzz.string import StringFuzzer
from fuzz.special import special_cases, corrected_keys
from sql_util.dbinfo import get_table_size, get_primary_keys, get_all_db_info_path, get_process_order
from sql_util.writedb import write_db_path
from sql_util.parse import extract_typed_value_in_comparison_from_query
from sql_util.value_typing import type_values_w_db
import os
from itertools import chain


process_count = multiprocessing.cpu_count()
number_dtype_str = ['int', 'float', 'double', 'decimal', 'numeric', 'number', 'real']
char_dtype_str = ['char', 'text', 'BLOB']
ADDITIVE_C = 200
MULTIPLICATIVE_C = 0
MIN_C = 30
QUERY_VALUES_WEIGHT = 30
ELEMENT_DROPOUT = 0.1


E = TypeVar('E')
def random_choices(l: List[E], k: int) -> List[E]:
    if len(l) == 0:
        return [None for _ in range(k)]
    idxes = [random.randint(0, len(l) - 1) for _ in range(k)]
    return [l[idx] for idx in idxes]


# the type in the database of spider is not very rigorous
# so things are a little slacky here.
def get_fuzzer_from_type_str(dtype_str: str, elements: List, p=0.5, max_l0=float('inf')) -> BaseFuzzer:
    #elements = [e for e in elements if random.random() > ELEMENT_DROPOUT]
    dtype_str = dtype_str.lower()
    if dtype_str == 'time':
        return TimeFuzzer(elements, p=p, max_l0=max_l0)
    if dtype_str == '' or dtype_str == 'blob':
        return StringFuzzer(elements, p=p, max_l0=max_l0)
    if 'date' in dtype_str or 'timestamp' in dtype_str:
        if len(elements) > 0 and isint(elements[0]):
            return NumberFuzzer(elements, p=p, max_l0=max_l0, scale=4)
        return DateFuzzer(elements, p=p, max_l0=max_l0)
    if 'bool' in dtype_str:
        return BoolFuzzer(elements, p=p)
    if 'bit' in dtype_str:
        return BitFuzzer(elements, p=p)
    if 'year' in dtype_str:
        return NumberFuzzer(elements, p=p, scale=4, max_l0=max_l0)
    unsigned = False
    if 'unsigned' in dtype_str:
        unsigned = True
    args = []
    if '(' in dtype_str:
        args = [int(x) for x in dtype_str[dtype_str.index('(') + 1:dtype_str.index(')')].split(',')]

    for s in number_dtype_str:
        if s in dtype_str:
            scale, precision = 10, 0
            if len(args) != 0:
                scale = args[0]
                if len(args) > 1:
                    scale -= args[1]
                    precision = args[1]
            is_int = 'int' in dtype_str
            return NumberFuzzer(elements, p, max_l0=max_l0, scale=scale, unsigned=unsigned, is_int=is_int, precision=precision)

    for s in char_dtype_str:
        if s in dtype_str:
            length = 20
            if len(args) != 0:
                length = args[0]
            return StringFuzzer(elements, p, max_l0=max_l0, length=length)


# given
# column2elements: a map from column name to the elements in the column
# primary_keys: a list of column names that are the primary key of the table
# return the filtered column2elements, without repetition in primary keys
def filter_by_primary(column2elements: Dict[str, List], primary_keys: List[str])\
        -> Dict[str, List]:
    if len(primary_keys) == 0:
        return column2elements
    num_elements = len(column2elements[primary_keys[0]])
    filtered_idx, existing_keys = set(), set()
    for idx in range(num_elements):
        key = tuple([column2elements[k][idx] for k in primary_keys])
        if key in existing_keys:
            filtered_idx.add(idx)
            continue
        existing_keys.add(key)

    filtered_column2elements = OrderedDict()
    for column_name, elements in column2elements.items():
        filtered_column2elements[column_name] = [elements[idx] for idx in range(num_elements) if idx not in filtered_idx]
    return filtered_column2elements


def filter_by_unique_keys(column2elements: Dict[str, List], unique_keys: Set[str])\
        -> Dict[str, List]:
    for k in unique_keys:
        column2elements = filter_by_primary(column2elements, [k])
    return column2elements


def restore_order(column2elements: Dict[str, List], column_order: List[str]) -> Dict[str, List]:
    result = OrderedDict()
    for column in column_order:
        result[column] = column2elements[column]
    return result


def rand_lin(x: int, min_c: int = MIN_C) -> int:
    return min_c + int(random.random() * ADDITIVE_C + x * (random.random() * MULTIPLICATIVE_C))


class DBFuzzer:

    def __init__(self, sqlite_path: str, tab_col2values: Dict[Tuple[str, str], List[str]], table_size: int, fuzzing: bool, p: float = 0.2):
        self.sqlite_path = sqlite_path
        self.values = tab_col2values
        self.table_column_properties, self.child2parent, self.table_column2elements = get_all_db_info_path(sqlite_path)
        self.orig_table_name2table_size = get_table_size(self.table_column2elements)
        self.table_column_order, self.table_order = get_process_order(self.child2parent, self.table_column_properties)
        self.table_primary_keys = get_primary_keys(self.table_column_properties)
        self.p = p
        self.fuzzers = OrderedDict()

        self.fuzzing = fuzzing
        self.table_size = table_size

        self.initialize_fuzzer()

    def get_fuzz(self) -> Dict[str, Dict[str, List]]:
        '''
        if table_size is None:
            if random.random() < 0.5:
                table_name2table_size = {table_name: rand_lin(v, MIN_C * random.random()) for table_name, v in self.orig_table_name2table_size.items()}
            else:
                table_name2table_size = {}
                cur_level_size = 1 + MIN_C * random.random()
                for tables in self.table_order:
                    for table_name in tables:
                        table_name2table_size[table_name] = cur_level_size
                    cur_level_size *= 4
                    cur_level_size = min(cur_level_size, 2500)
        '''
        table2column2elements = {}
        # print(f"table order:{self.table_order}")
        # print(f"len of table-order:{len(self.table_order)}")
        for idx, table_names in enumerate(self.table_order):
            # print(f"{idx}:{table_names}")
            for table_name in table_names:
                column2elements = {}
                table2column2elements[table_name] = column2elements
                primary_keys = self.table_primary_keys[table_name]

                # the column names in order
                # sometimes a column might refer to (foreign key) another column in the same table
                fuzz_order_column_names = []
                for table_columns in self.table_column_order:
                    for table_column in table_columns:
                        if table_column[0] == table_name:
                            fuzz_order_column_names.append(table_column[1])

                orig_order_column_names = []
                for t, column_name in self.table_column_properties:
                    if t == table_name:
                        orig_order_column_names.append(column_name)

                unique_keys = set([k for k in orig_order_column_names
                                   if self.table_column_properties[(table_name, k)]['unique']])

                # table_size = 1 + int(random.random() * table_name2table_size[table_name])
                for column_name in fuzz_order_column_names:
                    table_column = (table_name, column_name)
                    if table_column in self.child2parent:
                        parent_table, parent_column = self.child2parent[table_column]
                        assert parent_table in table2column2elements.keys(), "table %s should have been fuzzed" % parent_table
                        parent_elements = table2column2elements[parent_table][parent_column]
                        column2elements[column_name] = random.choices(parent_elements, k=self.table_size) if parent_elements != [] else [None for _ in range(self.table_size)]
                    else:
                        column_fuzzer = self.fuzzers[table_column]
                        # column2elements[column_name] = column_fuzzer.sample(self.fuzzing, self.table_size)
                        try:
                            samples = column_fuzzer.sample(self.fuzzing, self.table_size)
                            if len(samples) != self.table_size:
                                print(f"[WARNING] {table_column} returned {len(samples)} samples, padding with None")
                                samples += [None] * (self.table_size - len(samples))
                            
                            column2elements[column_name] = samples
                        except Exception as e:
                            print(f"[EXCEPTION] Sampling failed for {table_column}: {e}")
                            print(f"    Fuzzer type: {type(column_fuzzer)}")
                            print(f"    Fuzzer base elements: {getattr(column_fuzzer, 'base', None)}")
                            raise
                # filter by unique primary key constraint
                # and restore the original column order
                # and filter unique columns to avoid repitition
                transformations = [
                    (filter_by_primary, primary_keys),
                    (restore_order, orig_order_column_names),
                    (filter_by_unique_keys, unique_keys)
                ]
                for f, arg in transformations:
                    column2elements = f(column2elements, arg)
                table2column2elements[table_name] = column2elements
        return table2column2elements

    def initialize_fuzzer(self):
        #if random.random() < 0.3:
        #    self.p = 0
        for k in self.table_column_order[0]:
            dtype_str, elements = self.table_column_properties[k]['type'], self.table_column2elements[k]
            # 防止 None
            elements = elements or []
            query_values = self.values.get(k, [])  
            query_values = query_values or []      

            if self.fuzzing:
                sampled_elements = random.choices(elements, k=min(5, len(elements))) if elements else []
                elements = sampled_elements + list(set(query_values))
            else:
                elements = elements + list(set(query_values))
            

            elements = [e for e in elements if e is not None]
            # if not elements:
            #     dtype_str_lower = dtype_str.lower()
            #     if any(num_type in dtype_str_lower for num_type in number_dtype_str):
            #         elements = [random.uniform(0, 1000)]
            #     elif any(char_type in dtype_str_lower for char_type in char_dtype_str):
            #         elements = ["".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))]
            #     elif "bool" in dtype_str_lower:
            #         elements = [random.choice([True, False])]
            #     elif "date" in dtype_str_lower or "timestamp" in dtype_str_lower:
            #         from datetime import datetime, timedelta
            #         elements = [(datetime.now() - timedelta(days=random.randint(0, 3650))).date()]
            #     elif "time" in dtype_str_lower:
            #         elements = [f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"]
            #     else:
            #         elements = ["default"]
            #     print(f"[WARNING] No valid data for {k}, generated random element {elements}")

            if not elements:
                dtype_str_lower = dtype_str.lower()
                if any(num_type in dtype_str_lower for num_type in number_dtype_str):
                    elements = [random.uniform(0, 1000) for _ in range(self.table_size)]
                elif any(char_type in dtype_str_lower for char_type in char_dtype_str):
                    elements = ["".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5)) for _ in range(self.table_size)]
                elif "bool" in dtype_str_lower:
                    elements = [random.choice([True, False]) for _ in range(self.table_size)]
                elif "date" in dtype_str_lower or "timestamp" in dtype_str_lower:
                    from datetime import datetime, timedelta
                    elements = [(datetime.now() - timedelta(days=random.randint(0, 3650))).date() for _ in range(self.table_size)]
                elif "time" in dtype_str_lower:
                    elements = [f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}" for _ in range(self.table_size)]
                else:
                    elements = ["default"] * self.table_size
                print(f"[WARNING] No valid data for {k}, generated {len(elements)} fallback elements: {elements[:3]}...")


            checked = self.table_column_properties[k]['checked']
            p = 0 if checked else self.p
            special_key = (self.sqlite_path, k[0], k[1])
            if special_key in special_cases:
                self.fuzzers[k] = special_cases[special_key](elements, p)
            else:
                if special_key in corrected_keys:
                    dtype_str = corrected_keys[special_key]
                if random.random() < 0.5:
                    table_name, column_name = k
                    primary_keys = self.table_primary_keys[table_name]
                    max_l0 = random.random() * 30

                    if column_name in primary_keys:
                        if len(primary_keys) == 1:
                            max_l0 = float('inf')
                        if len(primary_keys) > 1:
                            max_l0 = 30
                    self.fuzzers[k] = get_fuzzer_from_type_str(dtype_str, elements, p, max_l0=max_l0)
                else:
                    self.fuzzers[k] = get_fuzzer_from_type_str(dtype_str, elements, p)



def generate_dbfuzzer(orig_path, queries, table_size, fuzzing=True):
    typed_values = list(chain(*[extract_typed_value_in_comparison_from_query(query) for query in queries]))
    tab_col2values = type_values_w_db(orig_path, typed_values, loose=False)
    dbfuzzer = DBFuzzer(orig_path, tab_col2values, table_size, fuzzing)
    return dbfuzzer

