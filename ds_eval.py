import json
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
from openai import OpenAI


# 系统提示和prompt模板保持与用户定义一致
system_msg = (
    "用户和助手之间的对话。用户提出一个问题，由助手来回答。助手首先在脑海中逐步思考推理过程，然后向用户提供答案。推理过程和答案分别用<思考> </思考>和<回答> </回答>标签括起来，即，"
    "<思考> 推理过程 </思考><回答> 答案 </回答>")

JEC_multi_choice_prompt = '''你是一名法学专家。现在请你解答司法考试中的一道选择题，请你找出所有正确的选项。每道题可能有一个或者多个正确答案。在解答之前，你需要先针对每个提供的选项给出详细的解释。你需要在回答的最后用大括号圈出给出的答案，例如"{{B}}"或者"{{ABD}}"。

问题：{question}

选项：
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}'''


def jec_multi_choice_prompt_template(entry: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    """根据用户定义的模板生成对话消息"""
    prompt = JEC_multi_choice_prompt.format(
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


def process_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """与用户定义一致的数据处理逻辑：过滤空答案样例"""
    processed_data = []
    for idx, example in enumerate(dataset):
        if not example.get('answer'):  # 检查答案是否为空
            print(f"Skipping example {idx} with empty answer: {example}")
            continue
        processed_data.append(example)
    return processed_data


class DeepseekEvaluator:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/beta"
        )
        self.total_tokens = 0
        self.max_tokens = 0

    def call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """调用DeepSeek API获取模型响应"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                temperature=0,
                max_tokens=1024
            )
            
            # 统计token使用
            usage = response.usage
            tokens = usage.total_tokens
            self.total_tokens += tokens
            self.max_tokens = max(self.max_tokens, tokens)

            return {
                "success": True,
                "content": response.choices[0].message.content,
                "tokens": tokens
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tokens": 0
            }

    def evaluate(self, dataset_path: str, output_dir: str = "results", test_samples: int = 1) -> None:
        """主评估流程"""
        # 加载并处理数据（与用户定义的process_dataset一致）
        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_dataset = json.load(f)
        processed_dataset = process_dataset(raw_dataset)  # 过滤空答案样例
        
        # 截取指定数量的测试样例（优先保证数据有效性）
        test_dataset = processed_dataset[:test_samples]

        results = []
        for item in tqdm(test_dataset, desc="Evaluating Samples"):
            # 使用用户定义的模板生成对话消息
            messages = jec_multi_choice_prompt_template(item, system_msg)

            # 调用API（无速率限制）
            response = self.call_api(messages)
            if not response["success"]:
                print(f"API请求失败：{response['error']}")
                continue

            # 评估响应结果
            eval_result = self.analyze_response(response["content"], item["answer"])

            # 构建结果记录
            results.append({
                **item,
                "model_response": response["content"],
                "tokens_used": response["tokens"],
                "metrics": {
                    "format_valid": self.check_format(response["content"]),
                    **eval_result
                }
            })

        # 保存结果
        self.save_output(results, output_dir)

    def check_format(self, response: str) -> float:
        """检查响应是否符合<思考><回答>格式"""
        return 1.0 if re.search(r"<思考>.*?</思考>\s*<回答>.*?</回答>", response, re.DOTALL) else 0.0

    def analyze_response(self, response: str, gt: List[str]) -> Dict[str, Any]:
        """分析模型响应与标准答案的匹配情况"""
        # 提取答案部分（与用户要求的格式一致）
        answer_match = re.search(r'\{\{?([A-Z]+)\}?\}', response)
        predicted = set(answer_match.group(1).upper()) if answer_match else set()
        gt_set = set(gt.upper() if isinstance(gt, str) else gt)

        # 计算评估指标
        tp = len(predicted & gt_set)
        fp = len(predicted - gt_set)
        fn = len(gt_set - predicted)
        return {
            "predicted": sorted(list(predicted)),
            "is_correct": predicted == gt_set,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "f1": 2 * (tp / (tp + fp) * tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))) if tp else 0.0
        }

    def save_output(self, results: List[Dict[str, Any]], output_dir: str) -> None:
        """保存完整结果和摘要报告"""
        Path(output_dir).mkdir(exist_ok=True)
        with open(Path(output_dir)/"full_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        summary = {
            "total_samples": len(results),
            "accuracy": sum(1 for r in results if r["metrics"]["is_correct"]) / len(results) if results else 0.0,
            "avg_precision": sum(r["metrics"]["precision"] for r in results) / len(results) if results else 0.0,
            "avg_recall": sum(r["metrics"]["recall"] for r in results) / len(results) if results else 0.0,
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
        }
        with open(Path(output_dir)/"summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 使用示例（需替换为有效API Key）
    evaluator = DeepseekEvaluator(api_key="sk-6d89aaa30ee347a7bf36102afa93bb04")
    evaluator.evaluate(
        dataset_path="/home/wangyc/verl/data/lawyer/jecqa/JEC_1_multi_choice_test.json",
        output_dir="evaluation_results",
        test_samples=1  # 测试1个样例
    )
