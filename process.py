import json
import re

def parse_sample(sample, idx):
    gpt_answer = sample.get("gpt_answer", "")
    info_answer = sample.get("info", {}).get("answer", [])

    # 提取答案：如 {D} 或 {AC}
    ans_match = re.search(r'\{[\u4e00-\u9fffA-Za-z,\s]*\}[^{]*$', gpt_answer)
    answers = list(sorted(set(re.findall(r'[A-Z]', ans_match.group())))) if ans_match else []
    gold = sorted(set(info_answer))

    return {
        "predicted": answers,
        "gold": gold,
        "correct": answers == gold
    }

def process(input_file, output_file):
    processed = []
    correct_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line)
                parsed = parse_sample(item, idx)
                processed.append(parsed)
                if parsed["correct"]:
                    correct_count += 1
            except Exception as e:
                print(f"解析失败 index={idx}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    total = len(processed)
    accuracy = correct_count / total if total > 0 else 0
    print(f"完成处理，共 {total} 条。")
    print(f"正确数量: {correct_count}")
    print(f"正确率: {accuracy:.2%}")
    print(f"结果保存在 {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", default="/home/wangyc/verl/JEC_1_multi_choice_train_output.jsonl", help="原始输入文件 (含instruction/output)")
    parser.add_argument("--out_file", default="tf.json", help="输出JSON文件")
    args = parser.parse_args()

    process(args.in_file, args.out_file)
