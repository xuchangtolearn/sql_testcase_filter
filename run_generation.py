import os
import json
import pickle
import vllm
from tqdm import tqdm
import argparse
import re
import gc
import sqlparse

# --- 阶段 0 所需的特定导入 ---
from fuzz.small_database_generation import generate_db_to_distinguish_queries as generate_small_db_for_step0

# --- 后续阶段所需的导入 ---
from fuzz.database_generation import generate_db_to_distinguish_queries
from fuzz.neighbor_sql_generation import generate_neighbor_queries_path
from sql_util.dbinfo import get_db_info_prompt
from utils.empirical_utils import get_all_used_tab_col_string_based, test_suite_judge_result_predict, my_get_sql_execution_result
from utils.opensource_model import llm_inference

# --- 共享变量和字典 ---
output_format_prompt = {
    'col_order': 'The output should be in the format of (table_name.column_name, list of the cell)',
    'row_order': 'The output should be several rows selected from the database or several aggregation results.'
}


# --- 阶段 0: 生成小型基础数据库的函数 ---
def db_generate_for_step0(sql, orig_db_path, new_db_root, data_index, data_type, db_type, test_case_num=1):
    """为阶段 0 定制的数据库生成函数。"""
    tmp = generate_neighbor_queries_path(orig_db_path, sql)
    neighbor_queries = []
    for neighbor in tmp:
        try:
            sqlparse.parse(neighbor)
            neighbor_queries.append(neighbor)
        except Exception as e:
            if repr(neighbor).strip():
                print(f"无效的邻居查询已被跳过: {repr(neighbor)}, 错误: {e}")
            continue
    # 调用专门用于生成小型基础数据库的模块
    db_path_list = generate_small_db_for_step0(sql, neighbor_queries, orig_db_path, new_db_root, data_index, data_type, db_type, test_case_num=test_case_num)
    return db_path_list

def step0_generate_base_databases(args):
    """阶段 0: 为特定数据库列表生成小型基础数据库。"""
    print("--- 阶段 0: 生成小型基础数据库 ---")
    data_type = args.data_type
    # 此步骤的 db_type 是固定的，因为它服务于特定的生成目的
    db_type = 'select_columns'
    test_case_num = 1

    # 步骤 1: 从 .json 文件加载数据，为每个 db_id 映射一个代表性的 SQL 查询
    print(f"从 {args.data_path} 加载 SQL 查询...")
    db_id_to_sql_map = {}
    with open(args.data_path, encoding='utf-8') as file:
        data = json.load(file)
    for entry in data:
        db_id = entry.get['db_id']
        # 在您的数据集中，SQL查询的键名可能是 'sql' 或 'SQL'
        sql = entry['sql']
        if db_id and sql and db_id not in db_id_to_sql_map:
            db_id_to_sql_map[db_id] = sql
    
    # 步骤 2: 扫描 --db_dir 目录，获取所有需要处理的 db_id
    if not os.path.isdir(args.db_dir):
        print(f"错误: 数据库目录 '{args.db_dir}' 不存在或不是一个目录。")
        return

    print(f"扫描数据库目录: {args.db_dir}...")
    try:
        all_db_ids = [d.name for d in os.scandir(args.db_dir) if d.is_dir()]
    except FileNotFoundError:
        print(f"错误: 找不到数据库目录 '{args.db_dir}'。")
        return

    print(f"找到 {len(all_db_ids)} 个数据库待处理。")

    # 步骤 3: 遍历所有找到的 db_id 并生成小型数据库
    for index, db_id in tqdm(enumerate(all_db_ids), total=len(all_db_ids), desc="生成基础数据库"):
        
        # 从映射中获取 SQL
        sql = db_id_to_sql_map.get(db_id)
        if not sql:
            print(f"警告：在 {args.data_path} 中未找到数据库 '{db_id}' 对应的 SQL 查询，跳过。")
            continue
            
        print(f"\n正在处理 db_id: {db_id}")
        
        # 原始（大型）数据库的路径
        orig_db_path = os.path.join(args.db_dir, db_id, f'{db_id}.sqlite')
        # 新生成的小型数据库的根目录
        small_db_base_dir = os.path.join(os.path.dirname(args.data_path), 'small_database')
        new_db_root = os.path.join(small_db_base_dir, db_id)

        if not os.path.exists(orig_db_path):
            print(f"警告：找不到原始数据库 {orig_db_path}，跳过 {db_id}")
            continue

        try:
            # 注意: 此处的 index 是新列表的索引，仅为满足函数签名要求
            db_generate_for_step0(sql, orig_db_path, new_db_root, index, data_type, db_type, test_case_num)
        except Exception as e:
            print(f"在为 db_id {db_id} 生成基础数据库时发生错误: {e}")
            continue
            
    print("--- 阶段 0 完成 ---")


# --- 后续阶段所需的函数 ---
def db_generate_fuzzing_variants(sql, orig_db_path, data_index, data_type, db_type, test_case_num):
    """为后续阶段生成模糊测试数据库变体。"""
    tmp = generate_neighbor_queries_path(orig_db_path, sql)
    neighbor_queries = []
    for neighbor in tmp:
        try:
            sqlparse.parse(neighbor)
            neighbor_queries.append(neighbor)
        except Exception as e:
            if repr(neighbor).strip():
                print(f"无效的邻居查询已被跳过: {repr(neighbor)}, 错误: {e}")
            continue
    # 调用用于生成模糊测试变体的模块
    db_path_list = generate_db_to_distinguish_queries(sql, neighbor_queries, orig_db_path, data_index, data_type, db_type, test_case_num)
    return db_path_list

def generate_test_case(prompts, original_infos, batch_size=30000, model_path="/mnt/public/data/lh/models/Qwen2.5-Math-7B-Instruct"):
    save_data = []
    print(f"Total prompts to predict: {len(prompts)}")

    llm = vllm.LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()

    responses = []
    # print(f"prompts[0]:{prompts[0]}")
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batch LLM Inference"):
        sub_prompts = prompts[i:i+batch_size]
        sub_responses = llm_inference(sub_prompts, llm, tokenizer)
        responses.extend(sub_responses)

    del llm
    del tokenizer
    gc.collect()

    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    test_suite_by_data = {}

    assert len(responses) == len(original_infos)

    for idx, info in enumerate(original_infos):
        db_path = info['db_path']
        question = info['question']
        sql = info['sql']
        db_id = info['db_id']
        data_index = info['data_index']

        # predict_result = process_response(responses[idx])
        # print(f"response:{responses[idx]}")
        predict_result = responses[idx]

        try:
            execution_result, _ = my_get_sql_execution_result(sql, db_path, format='row_order')
        except Exception as e:
            print(f"Execution error at db: {db_path}", e)
            execution_result = []
        
        # print(f"data_index:{data_index}")
        # print(f"predict_result:{predict_result}")
        # print(f"execution_result:{execution_result}")

        is_equal = test_suite_judge_result_predict(predict_result, execution_result, False)

        if data_index not in test_suite_by_data:
            test_suite_by_data[data_index] = []

        if is_equal:
            # print(f"Data {data_index} --- true")
            test_suite_by_data[data_index].append((db_path, predict_result, execution_result, True))
        else:
            # print(f"Data {data_index} --- false")
            test_suite_by_data[data_index].append((db_path, predict_result, execution_result, False))
            
        # save_data.append({'data_entry_id': data_index, 'test_suites': list(test_suite_by_data[data_index])})

        # test_suite_by_data[data_index].append((db_path, predict_result, execution_result, is_equal))

    for index in range(max(info['data_index'] for info in original_infos) + 1):
        test_suite = test_suite_by_data.get(index, [])
        save_data.append({'data_entry_id': index, 'test_suites': list(test_suite)})

    return save_data

def get_model_tag(model_path):
    return os.path.basename(model_path).replace('/', '_').replace('\\', '_')

def build_all_prompts(batch_entries, output_format='row_order', db_format='csv'):
    prompts, original_infos = [], []
    for entry in tqdm(batch_entries, desc="构建提示"):
        # try:
        tab_col_used = get_all_used_tab_col_string_based(entry['sql'], db_path=entry['db_path'], db_id=entry['db_id'])
        db_info = get_db_info_prompt(entry['db_path'], table_col_list=tab_col_used, format=db_format)
        query_prompt = f"#Input natural language question: {entry['question']}\n#Input database:\n{db_info}\n#Output:"
        final_prompt = f"""### Given a database and a natural language question, please select the cells of the database or get aggregation results.\n### {output_format_prompt[output_format]}\n\n{query_prompt}"""
        # print(f"final_prompt: {final_prompt}")
        prompts.append(final_prompt)
        original_infos.append(entry)
        # except Exception as e:
        #     print(f"构建提示时出错: {e}, 问题: {entry.get('question', 'N/A')}")
        #     continue
    return prompts, original_infos

def filter_and_save_dataset(test_suite_results, original_data_path, final_output_path, threshold=0.6):
    print("\n--- 阶段 3: 根据评估结果筛选数据集 ---")
    if not os.path.exists(original_data_path):
        print(f"错误：找不到原始数据文件: {original_data_path}")
        return
    passed_entry_ids = set()
    for entry in test_suite_results:
        test_suites = entry["test_suites"]
        if not test_suites: continue
        consistency_true_count = sum(1 for ts in test_suites if ts[3] is True)
        if consistency_true_count >= len(test_suites) * threshold:
            passed_entry_ids.add(entry["data_entry_id"])
    print(f"筛选后满足条件（一致性>={threshold*100}%）的数据条数: {len(passed_entry_ids)}")
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    filtered_data = [entry for i, entry in enumerate(original_data) if i in passed_entry_ids]
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    print(f"原始数据集大小: {len(original_data)}")
    print(f"最终筛选出的数据集大小: {len(filtered_data)}")
    print(f"筛选后的数据已保存至: {final_output_path}")

# --- 主程序入口 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成、评估并筛选测试用例的完整流程。")
    parser.add_argument('--model_path', type=str, required=True, help='推理模型的路径')
    parser.add_argument('--db_dir', type=str, required=True, help='数据库目录')
    parser.add_argument('--data_path', type=str, required=True, help='用于最终筛选的原始数据集路径 (例如: bird_ori.json)')
    parser.add_argument('--output_path', type=str, required=True, help='筛选后最终数据集的输出路径 (例如: bird_filtered.json)')
    parser.add_argument('--threshold', type=float, default=0.6, help='筛选阈值')
    parser.add_argument('--data_type', type=str, default='train', choices=['train', 'dev'], help='数据类型 (例如: train/dev)')
    parser.add_argument('--db_type', type=str, default='select_values', help='数据库模糊测试用例生成类型')
    parser.add_argument('--test_case_num', type=int, default=10, help='为每个查询生成的模糊测试变体数量')
    args = parser.parse_args()
    output_dir = os.path.dirname(args.data_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 阶段 0: 生成小型基础数据库 ---
    # step0_generate_base_databases(args)

    # --- 阶段 1: 生成模糊测试提示 ---
    print("\n--- 阶段 1: 生成模糊测试提示 ---")
    source_data_path = args.data_path
    with open(source_data_path, encoding='utf-8') as file:
        data = json.load(file)

    all_batch_entries = []
    # 在这个阶段，我们跳过那些在阶段0已经处理过的困难数据库
    small_db_base_dir = os.path.join(os.path.dirname(args.data_path), 'small_database')
    for index, entry in tqdm(enumerate(data), total=len(data), desc="生成提示"):        
        # 使用阶段0生成的小型数据库作为基础
        orig_db_path = os.path.join(small_db_base_dir, entry['db_id'], 'select_columns_test_case', f"{entry['db_id']}.db")
        if not os.path.exists(orig_db_path):
            # print(f"警告：未找到小型数据库 {orig_db_path}，跳过此条目。")
            orig_db_path = os.path.join(args.db_dir, entry['db_id'], f"{entry['db_id']}.sqlite")
            print(f"使用原始数据库 {orig_db_path} 作为基础。")
            
        # try:
        db_path_list = db_generate_fuzzing_variants(entry['sql'], orig_db_path, index, args.data_type, args.db_type, args.test_case_num)
        for db_path in db_path_list:
            all_batch_entries.append({
                'question': entry['question'], 'sql': entry['sql'], 'db_id': entry['db_id'],
                'db_path': db_path, 'data_index': index,
            })
        # except Exception as e:
        #     print(f"为数据索引 {index} 生成提示时出错: {e}")
        #     continue
    
    print(f"共为测试生成了 {len(all_batch_entries)} 个数据库条目。")
    prompts, original_infos = build_all_prompts(all_batch_entries)
    prompt_infos = {
        'prompts': prompts,
        'original_infos': original_infos
    }
    prompt_infos_path = os.path.join(output_dir, 'prompts_and_infos.json')
    with open(prompt_infos_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_infos, f, indent=2, ensure_ascii=False)
    print(f"已将提示和原始信息保存到 'prompts_and_infos.json' 文件中。")

    with open(prompt_infos_path, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    prompts = prompt_data['prompts']
    original_infos = prompt_data['original_infos']
    if not prompts:
        print("未能生成任何提示。后续阶段将跳过。")
    else:
        # --- 阶段 2: 使用 LLM 生成并评估测试用例 ---
        print("\n--- 阶段 2: 使用 LLM 生成并评估测试用例 ---")
        save_data = generate_test_case(prompts, original_infos, model_path=args.model_path)
        model_tag = get_model_tag(args.model_path)
        test_suite_output_path = os.path.join(output_dir, f'{args.data_type}_{args.db_type}_test_suite_{model_tag}.pkl')
        with open(test_suite_output_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"\n完整的测试套件评估结果已保存至: {test_suite_output_path}")

        # --- 阶段 3: 根据评估结果筛选原始数据集 ---
        filter_and_save_dataset(
            test_suite_results=save_data,
            original_data_path=args.data_path,
            final_output_path=args.output_path
        )

    print("\n所有流程完成！")


# python run_generation.py --model_path /mnt/public/data/lh/data/xc/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/ckpts/checkpoint-10275 --db_dir data/database --data_path data/train_test.json --output_path data/output_test.json > output_test.txt