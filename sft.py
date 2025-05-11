import os
import json
import datasets
from tqdm import tqdm

def validate_data_item(item):
    """校验数据格式"""
    required_fields = ['instruction', 'output']
    for field in required_fields:
        if field not in item:
            raise KeyError(f"Missing required field: {field}")
        if not isinstance(item[field], str):
            raise TypeError(f"Field {field} should be string type, got {type(item[field])}")

def convert_to_sft_format(input_path, output_path, dataset_type='train'):
    """将原始JSON转换为SFT训练需要的parquet格式"""
    
    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {input_path}: {str(e)}")

    processed = []
    error_count = 0

    # 添加进度条
    with tqdm(total=len(raw_data), desc=f"Processing {dataset_type} data") as pbar:
        for idx, item in enumerate(raw_data):
            try:
                # 数据校验
                validate_data_item(item)

                # 构造训练样本
                processed.append({
                    "instruction": str(item['instruction']).strip(),  # 强制转换为字符串并去除空白
                    "output": str(item['output']).strip(),
                    "extra_info": {
                        "idx": idx,
                        "data_source": "jec-qa-sft",
                        "answer_type": item.get("answer_type", "multi_choice")  # 添加默认值
                    }
                })
            except Exception as e:
                error_count += 1
                print(f"Error processing item {idx}: {str(e)}")
            finally:
                pbar.update(1)

    # 数据质量报告
    print(f"\nData processing complete for {dataset_type}")
    print(f"Total samples: {len(raw_data)}")
    print(f"Successfully processed: {len(processed)}")
    print(f"Failed samples: {error_count}")

    # 转换为Dataset并保存
    if len(processed) > 0:
        dataset = datasets.Dataset.from_list(processed)
        
        # 添加数据集特征描述
        dataset_features = datasets.Features({
            "instruction": datasets.Value("string"),
            "output": datasets.Value("string"),
            "extra_info": datasets.Features({
                "idx": datasets.Value("int64"),
                "data_source": datasets.Value("string"),
                "answer_type": datasets.Value("string")
            })
        })
        dataset = dataset.cast(dataset_features)
        
        dataset.to_parquet(output_path)
        print(f"Saved {len(dataset)} samples to {output_path}")
    else:
        raise ValueError("No valid data processed, check input file format")

if __name__ == "__main__":
    # 配置路径（请根据实际路径修改）
    config = {
        "train_json": "/home/wangyc/verl/jec_distill_r1_multi-choice.json",
        "test_json": "/home/wangyc/verl/jec_distill_r1_multi-choice.json",  # 建议使用不同的测试集文件
        "output_dir": "/home/wangyc/verl/processed_data"
    }

    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)

    # 处理训练集
    print("\n" + "="*50 + " Processing TRAIN Data " + "="*50)
    convert_to_sft_format(
        input_path=config['train_json'],
        output_path=os.path.join(config['output_dir'], "train.parquet"),
        dataset_type='train'
    )

    # 处理测试集（建议使用不同的文件）
    print("\n" + "="*50 + " Processing TEST Data " + "="*50)
    convert_to_sft_format(
        input_path=config['test_json'],
        output_path=os.path.join(config['output_dir'], "test.parquet"),
        dataset_type='test'
    )

    # 最终验证
    print("\n" + "="*50 + " Final Validation " + "="*50)
    try:
        train_set = datasets.load_dataset("parquet", data_files=os.path.join(config['output_dir'], "train.parquet"))
        print("\nTrain set sample:")
        print(train_set['train'][0])
    except Exception as e:
        print(f"Train set validation failed: {str(e)}")

    try:
        test_set = datasets.load_dataset("parquet", data_files=os.path.join(config['output_dir'], "test.parquet"))
        print("\nTest set sample:")
        print(test_set['train'][0])
    except Exception as e:
        print(f"Test set validation failed: {str(e)}")