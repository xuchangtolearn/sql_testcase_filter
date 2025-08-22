import pickle
import json
import os
import pandas as pd

def load_pkl_file(pkl_path):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"文件不存在: {pkl_path}")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        if not data:
            raise ValueError("文件内容为空")
        return data
    except Exception as e:
        raise RuntimeError(f"加载 pkl 文件失败: {e}")

def convert_to_readable_json(data, output_path):
    json_data = []
    for item in data:
        try:
            data_id = item.get('data_entry_id', None)
            test_suites = item.get('test_suites', [])
            suite_list = []
            for suite in test_suites:
                try:
                    db_path, predict_result, execution_result, consistency = suite
                    suite_list.append({
                        'db_path': db_path,
                        'predict_result': predict_result,
                        'execution_result': execution_result,
                        'consistency': consistency
                    })
                except Exception as e:
                    suite_list.append({'error': f'无法解析某个 test suite: {e}'})
            json_data.append({'data_entry_id': data_id, 'test_suites': suite_list})
        except Exception as e:
            json_data.append({'error': f'解析失败: {e}'})

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

def convert_to_csv(data, output_path):
    rows = []
    for item in data:
        data_id = item.get('data_entry_id', None)
        test_suites = item.get('test_suites', [])
        for suite in test_suites:
            try:
                db_path, predict_result, execution_result, consistency = suite
                rows.append({
                    'data_entry_id': data_id,
                    'db_path': db_path,
                    'predict_result': str(predict_result),
                    'execution_result': str(execution_result),
                    'consistency': consistency
                })
            except Exception as e:
                rows.append({
                    'data_entry_id': data_id,
                    'db_path': None,
                    'predict_result': None,
                    'execution_result': None,
                    'consistency': None,
                    'error': str(e)
                })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    pkl_file_path = 'train_select_values_test_suite_checkpoint-10275.pkl'
    json_output_path = 'train_select_values_test_suite_checkpoint-10275.json'
    csv_output_path = 'train_select_values_test_suite_checkpoint-10275.csv'

    try:
        data = load_pkl_file(pkl_file_path)
        convert_to_readable_json(data, json_output_path)
        convert_to_csv(data, csv_output_path)
        print(f"✅ 数据已成功保存为：\nJSON: {json_output_path}\nCSV: {csv_output_path}")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
