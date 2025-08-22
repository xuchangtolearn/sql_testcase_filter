import openai
import os
from time import sleep
import json
import requests
import vllm
from vllm import SamplingParams


def llm_inference(input_prompt, llm, tokenizer):
    """
    给定一组 prompts，使用 vLLM 模型批量推理并返回所有生成的文本答案。

    Args:
        input_prompt (List[str]): 输入的一组 prompt
        llm: vLLM LLM实例
        tokenizer: vLLM tokenizer实例

    Returns:
        List[str]: 每个 prompt 对应的生成文本
    """
    prompts = []
    for entry in input_prompt:
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': entry}],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(text)

    print(f"finished prompts: {len(prompts)}")

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0,                # 输出确定性
        max_tokens=4096,              # 限制最大生成长度
        stop_token_ids=[
            tokenizer.eos_token_id, 
            tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]                             # 提前终止输出
    )

    # 调用 vLLM 批量生成
    outputs = llm.generate(prompts, sampling_params ,use_tqdm=True)

    # 提取每条输出的 response text
    res_data = []
    for output in outputs:
        # 取每个 prompt 的第一条输出
        if output.outputs:
            response = output.outputs[0].text.strip()
            res_data.append(response)
        else:
            # 保险：如果output没有生成内容，填空字符串
            res_data.append("")

    return res_data




