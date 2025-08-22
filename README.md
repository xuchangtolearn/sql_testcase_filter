# sql_testcase_filter


通过以下命令执行完整的测试用例生成、评估和筛选流程。

### 运行命令示例

```bash
python run_generation.py \
    --model_path /mnt/public/data/lh/data/xc/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/ckpts/checkpoint-10275 \
    --db_dir data/database \
    --data_path data/train_test.json \
    --output_path data/output_test.json > output_test.txt
```

### 参数说明

  - `--model_path`: 指定用于推理的大模型的路径。
  - `--db_dir`: 存放所有 SQLite 数据库文件的根目录。
  - `--data_path`: 原始输入数据文件的路径，包含问题、SQL 查询和数据库 ID。
  - `--output_path`: 经过筛选后，最终输出的数据集文件的保存路径。
