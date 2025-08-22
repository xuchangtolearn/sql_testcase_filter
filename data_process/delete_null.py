import json
import random

# 读取 JSON 文件
with open('/mnt/public/data/lh/data/xc/sql_test_case_generation/data/train_select_values_sft_data.json', 'r') as file:
    data = json.load(file)

# 过滤出 "output" 不为空的条目
non_empty_entries = [entry for entry in data if entry['output'] == ""]

# 随机删除一半 "output" 不为空的条目
half_to_delete = len(non_empty_entries) // 2
entries_to_delete = random.sample(non_empty_entries, half_to_delete)

# 从原数据中删除这些条目
data = [entry for entry in data if entry not in entries_to_delete]

# 将修改后的数据保存到新文件
print(f"剩余数据量：{len(data)}")
with open('/mnt/public/data/lh/data/xc/sql_test_case_generation/data/testcase_select_values.json', 'w') as file:
    json.dump(data, file, indent=4)

print(f"已随机删除 {half_to_delete} 条 'output' 不为空的条目。")
