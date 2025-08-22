import sqlite3
import os
from tqdm import tqdm

def find_all_sqlite_files(base_path):
    sqlite_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.sqlite'):
                sqlite_files.append(os.path.join(root, file))
    return sqlite_files

def get_all_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"[ERROR] Failed to list tables in {db_path}: {e}")
        tables = []
    cursor.close()
    conn.close()
    return tables

def get_table_columns(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = [row[1] for row in cursor.fetchall()]
    except Exception as e:
        print(f"[ERROR] Failed to get columns for {table_name} in {db_path}: {e}")
        columns = []
    cursor.close()
    conn.close()
    return columns

def check_nulls_in_table(db_path, table_name, columns):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    null_columns = []
    for col in columns:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM '{table_name}' WHERE {col} IS NULL")
            count = cursor.fetchone()[0]
            if count > 0:
                null_columns.append((col, count))
        except Exception as e:
            print(f"[WARNING] Error checking NULLs in {table_name}.{col} of {db_path}: {e}")
            continue
    cursor.close()
    conn.close()
    return null_columns

def check_all_databases(base_dir):
    sqlite_files = find_all_sqlite_files(base_dir)
    
    for db_path in tqdm(sqlite_files):
        tables = get_all_tables(db_path)
        if not tables:
            continue

        db_has_null = False
        for table in tables:
            columns = get_table_columns(db_path, table)
            if not columns:
                continue

            null_columns = check_nulls_in_table(db_path, table, columns)
            if null_columns:
                if not db_has_null:
                    print(f"\nDatabase: {db_path}")
                    db_has_null = True
                print(f"  Table: {table}")
                for col, null_count in null_columns:
                    print(f"    Column '{col}' has {null_count} NULL values.")


if __name__ == "__main__":
    base_dir = "/mnt/public/data/lh/data/xc/dataset/OmniSQL-datasets/data/spider/database"
    check_all_databases(base_dir)
