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
    """
    # Find all answers in the format of {A, B, C, D, ...}
    match = re.search(r'\{[\u4e00-\u9fffA-Za-z,\s]*\}[^{]*$', predict)
    
    if match is None:
        return 0.0  # No answers found, return 0 score

    # Extract all the chosen answers (A, B, C, D, etc.)
    chosen_answers = set(re.findall(r'[A-Z]', match.group()))
    ground_truth = set(ground_truth)
    
    # Calculate penalties
    incorrect_answers = chosen_answers - ground_truth  # Chosen answers not in ground truth
    missing_answers = ground_truth - chosen_answers  # Correct answers missing from the prediction

    # Soft penalty: subtract 0.5 for each incorrect or missing answer
    score = max(0.0, 1.0 - 0.9 * (len(incorrect_answers) + len(missing_answers)))
    return score


def compute_score(predict, ground_truth, **kwargs) -> dict:
    """Compute the score of the completion."""
    format_score = format_reward(predict)
    accuracy_score = verify_multiple_choice(predict, ground_truth)
    return {
        "score": format_score + accuracy_score,
        "extra_info": {
            "format_reward": format_score,
            "answer_reward": accuracy_score,
        }
    }
