import os
import datasets
import json
from verl.utils.hdfs_io import copy, makedirs
import argparse
from typing import List, Dict, Any
from pathlib import Path


# 系统提示词
SYSTEM_MSG = (
    "用户和助手之间的对话。用户提出一个问题，由助手来回答。助手首先在脑海中逐步思考推理过程，"
    "然后向用户提供答案。推理过程和答案分别用<思考> </思考>和<回答> </回答>标签括起来，即，"
    "<思考> 推理过程 </思考><回答> 答案 </回答>"
)

# 多选题提示词模板
JEC_MULTI_CHOICE_PROMPT = '''你是一名法学专家。现在请你解答司法考试中的一道选择题，请你找出所有正确的选项。
每道题可能有一个或者多个正确答案。在解答之前，你需要先针对每个提供的选项给出详细的解释。
你需要在回答的最后用大括号圈出给出的答案，例如"{{B}}"或者"{{ABD}}"。

问题：{question}

选项：
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}'''


def jec_multi_choice_prompt_template(entry: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    """
    生成多选题的提示词模板
    
    Args:
        entry: 包含问题和选项的字典
        system_prompt: 系统提示词
    
    Returns:
        包含角色和内容的提示词列表
    """
    prompt = JEC_MULTI_CHOICE_PROMPT.format(
        question=entry['statement'],
        option_a=entry['option_list']["A"],
        option_b=entry['option_list']["B"],
        option_c=entry['option_list']["C"],
        option_d=entry['option_list']["D"]
    )
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]


def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    处理单个样本
    
    Args:
        example: 输入样本
        idx: 样本索引
    
    Returns:
        处理后的样本数据
    """
    return {
        "prompt": jec_multi_choice_prompt_template(example, SYSTEM_MSG),
        "data_source": "jec-qa-1-multi-choice",
        "reward_model": {
            "style": "rule",
            "ground_truth": example['answer'],
        },
        "extra_info": {
            'idx': idx
        }
    }


def process_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    处理整个数据集
    
    Args:
        dataset: 输入数据集
    
    Returns:
        处理后的数据集
    """
    data = []
    total_count = len(dataset)
    skipped_count = 0
    
    for idx, example in enumerate(dataset):
        if not example['answer']:
            print(f"跳过样本 {idx}: 答案为空")
            skipped_count += 1
            continue
        data.append(process_fn(example, idx))
    
    print(f"数据集统计信息:")
    print(f"- 总样本数: {total_count}")
    print(f"- 跳过样本数: {skipped_count}")
    print(f"- 最终样本数: {len(data)}")
    
    return data


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    加载JSON文件
    
    Args:
        file_path: JSON文件路径
    
    Returns:
        加载的数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='处理JEC多选题数据集')
    parser.add_argument('--local_dir', 
                       default='verl/data/jec-qa-1-multi-choice',
                       type=str,
                       help='输出目录路径')
    args = parser.parse_args()

    # 加载数据集
    train_dataset = load_json_file('data/lawyer/jecqa/JEC_1_multi_choice_train.json')
    test_dataset = load_json_file('data/lawyer/jecqa/JEC_1_multi_choice_test.json')

    # 处理数据集
    train_data = process_dataset(train_dataset)
    test_data = process_dataset(test_dataset)

    # 转换为Dataset对象
    train_data = datasets.Dataset.from_list(train_data)
    test_data = datasets.Dataset.from_list(test_data)

    # 创建输出目录
    output_dir = Path(args.local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存处理后的数据
    train_data.to_parquet(output_dir / 'train.parquet')
    test_data.to_parquet(output_dir / 'test.parquet')


if __name__ == '__main__':
    main()