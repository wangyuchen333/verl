import re


def format_reward(predict):
    """Reward function that checks if the completion has a specific format."""
    pattern = r".*?<思考>.*?</思考>.*?<回答>.*?</回答>"
    match = re.match(pattern, predict, re.DOTALL | re.MULTILINE) 
    return 1.0 if match else 0.0


def verify_multiple_choice(predict, ground_truth):
    """
    Soft Reward for multiple-choice questions:
    - Penalize 0.5 points for each incorrect answer.
    - Penalize 0.5 points for missing a correct answer.
    - Minimum score of 0.
    Also compute precision, recall, and F1.
    """
    match = re.search(r'\{[\u4e00-\u9fffA-Za-z,\s]*\}[^{]*$', predict)
    
    if match is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    chosen_answers = set(re.findall(r'[A-Z]', match.group()))
    ground_truth = set(ground_truth)
    
    accuracy = 1.0 if chosen_answers == ground_truth else 0.0

    incorrect_answers = chosen_answers - ground_truth
    missing_answers = ground_truth - chosen_answers

    score = max(0.0, 1.0 - 0.5 * (len(incorrect_answers) + len(missing_answers)))

    correct = len(chosen_answers & ground_truth)
    precision = correct / len(chosen_answers) if chosen_answers else 0.0
    recall = correct / len(ground_truth) if ground_truth else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return score, accuracy, precision, recall, f1


def compute_score(predict, ground_truth, **kwargs) -> dict:
    """Compute the score of the completion."""
    format_score = format_reward(predict)
    answer_score, accuracy, precision, recall, f1 = verify_multiple_choice(predict, ground_truth)
    return {
        "score": format_score + answer_score,
        "extra_info": {
            "format_reward": format_score,
            "answer_reward": answer_score,
            "acc": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    }
