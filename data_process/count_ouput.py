import json

def count_non_empty_outputs(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"len of data:{len(data)}")
    count = sum(1 for item in data if item.get('output', '') != '')
    return count

# 示例用法
json_path = '/mnt/public/data/lh/data/xc/sql_test_case_generation/data/train_select_values_sft_data.json' 
result = count_non_empty_outputs(json_path)
print(f'output不为空的项数: {result}')
