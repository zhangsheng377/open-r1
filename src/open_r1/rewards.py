"""Reward functions for GRPO training."""

import json
import math
import re
from collections import deque
from typing import Dict

import regex
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import is_e2b_available

if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import Sandbox

    load_dotenv()


def extract_answer_from_output(output):
    # 使用 re.finditer() 查找所有匹配项
    matches = list(re.finditer(r'<answer>(.*?)</answer>', output, re.DOTALL))
    if matches:
        # 获取最后一个匹配项的内容
        return matches[-1].group(1).strip()  # group(1) 是捕获的第一个组
    return ""


def extract_answer_from_label(label):
    pattern = r'\\boxed\{((?:[^{}]|(?R))*)\}'
    # 查找所有匹配的 \box{...}
    matches = regex.findall(pattern, label)
    if matches:
        # 获取最后一个匹配项的内容
        return matches[-1]
    return ""


def compute_cosine_similarity(text1, text2):
    try:
        vectorizer = CountVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0, 1]
    except:
        return 0


def accuracy_reward_function(model_output, label, similarity_threshold=0.8):
    model_answer = extract_answer_from_output(model_output)
    reference_answer = extract_answer_from_label(label)
    if model_answer and reference_answer:
        # 检查模型输出是否包含参考答案
        if reference_answer in model_answer:
            # print(
            #     f"accuracy_reward_function : model_answer : {model_answer} reference_answer : {reference_answer} score : 1")
            return 1  # 正奖励
        similarity = compute_cosine_similarity(model_answer, reference_answer)
        # print(
        #     f"accuracy_reward_function : model_answer : {model_answer} reference_answer : {reference_answer} similarity : {similarity}")
        if similarity >= similarity_threshold:
            return 1  # 正奖励
        return similarity
    # print(f"accuracy_reward_function : model_answer : {model_answer} reference_answer : {reference_answer} score : 0")
    return 0


def my_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        reward = accuracy_reward_function(model_output=content, label=sol)
        rewards.append(reward)
    return rewards


def my_ref_model_accuracy_reward(completions, solution, **kwargs):
    PROMPT = """你是一个判别推理正确性的专家。




# 问题
[问题]
{question}
[/问题]




# 参考答案
<参考答案>
{sol}
</参考答案>




# 模型回答
<模型回答>
{completion}
</模型回答>




# 你的任务目标
你的任务目标：根据给定的[问题]和[参考答案]，评估[模型回答]的正确性。请你不要回答问题，只需要关注[模型回答]最后的答案是否正确即可，不用关心答案前的推理过程。打分范围为0-10，正确为10分，错误为0分。请直接给出0-10的数字，且只需输出数字即可，不要有其他任何文字。"""
    ref_model_inference_func = kwargs["ref_model_inference_func"]
    problems = kwargs["problem"]
    contents = [completion[0]["content"] for completion in completions]
    contents = [extract_answer_from_output(content) for content in contents]
    # solutions = [extract_answer_from_output(sol) for sol in solution]
    # solutions = [extract_answer_from_label(sol) for sol in solution]
    print(f"problem : {problems[0]}")
    print(f"content : {contents[0]}")
    print(f"sol : {solution[0]}")
    input_texts = [PROMPT.format(question=problem, sol=sol, completion=content) for problem, content, sol in
                   zip(problems, contents, solution)]
    scores = ref_model_inference_func(input_texts)
    rewards = []
    for i, score in enumerate(scores):
        try:
            reward = float(score)
        except Exception as e:
            match = re.search(r'\d+', score)
            if match:
                reward = float(match.group(0))  # 返回匹配到的第一个数字
            else:
                print(f"!!! ref_model_inference_func return error!!! score : {score} contents : {contents[i]}")
                reward = 0
        reward = reward / 10.0
        if 0 > reward or 1 < reward:
            print(f"!!! ref_model_inference_func reward error!!! score : {score}")
            reward = 0
        rewards.append(reward)
    print(f"my_ref_model_accuracy_reward : rewards : {rewards}")
    return rewards


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        # print(f"accuracy_reward : content : {content} sol : {sol} reward : {reward}")
        # print(f"accuracy_reward : content : {content} gold_parsed : {gold_parsed} reward : {reward}")
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def my_ref_model_reasoning_steps_reward(completions, **kwargs):
    PROMPT = """你是一个判别推理过程性的专家。




# 模型回答
<模型回答>
{completion}
</模型回答>




# 你的任务目标
你的任务目标：评估[模型回答]的推理过程的好坏。一个理想的推理过程应该先分析、规划，再逐步分析，最后得出答案后还要再次检查，直到确认无误后再给出最后的回答。推理过程的关键在于有理有据，足够细致。你的打分范围为0-10，好为10分，坏为0分。请直接给出0-10的数字，且只需输出数字即可，不要有其他任何文字。"""
    ref_model_inference_func = kwargs["ref_model_inference_func"]
    contents = [completion[0]["content"] for completion in completions]

    input_texts = [PROMPT.format(completion=content) for content in contents]
    scores = ref_model_inference_func(input_texts)
    rewards = []
    for i, score in enumerate(scores):
        try:
            reward = float(score)
        except Exception as e:
            match = re.search(r'\d+', score)
            if match:
                reward = float(match.group(0))  # 返回匹配到的第一个数字
            else:
                print(
                    f"!!! my_ref_model_reasoning_steps_reward return error!!! score : {score} contents : {contents[i]}")
                reward = 0
        reward = reward / 10.0
        if 0 > reward or 1 < reward:
            print(f"!!! my_ref_model_reasoning_steps_reward reward error!!! score : {score}")
            reward = 0
        rewards.append(reward)
    print(f"my_ref_model_reasoning_steps_reward : rewards : {rewards}")
    return rewards


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


class DynamicMeanDeque:
    def __init__(self, maxlen):
        self.queue = deque(maxlen=maxlen)
        self.total = 0  # 维护累计和

    def append(self, value):
        if len(self.queue) == self.queue.maxlen:
            # 如果队列已满，减去最早被移除的元素
            self.total -= self.queue[0]
        self.queue.append(value)
        self.total += value

    def mean(self):
        if self.queue:
            return self.total / len(self.queue)
        return 0  # 如果队列为空，返回 0


def get_my_length_reward(
        target_len: int = 640,
):
    # dynamic_queue = DynamicMeanDeque(maxlen=100)

    def my_length_reward(completions, solution, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        completion_ids = kwargs["completion_ids"]
        rewards = []

        for content, sol, completion_ids_ in zip(contents, solution, completion_ids):
            gen_len = len(completion_ids_)
            # dynamic_queue.append(gen_len)
            # target_len_ = dynamic_queue.mean() * 2  # 一开始可能会振荡，后面就稳定了
            # if target_len_ > target_len:
            #     target_len_ = target_len
            progress = gen_len / max(target_len, 1) if gen_len < target_len else 1.0
            # is_correct = accuracy_reward_function(model_output=content, label=sol)

            x = progress
            # reward = x * (x * (x - 3) + 3)
            # reward = x * (2 - x)
            # reward = (math.cos(x * math.pi) + 1) / 2
            # reward = 1 - (math.cos(x * math.pi) + 1) / 2
            # reward = x * x
            reward = x
            # if not is_correct:
            #     reward = -reward
            #
            # if not is_correct:
            #     reward = reward / 2 - 1

            rewards.append(float(reward))
        return rewards

    return my_length_reward


def my_get_cosine_scaled_reward(
        min_value_wrong: float = -1.0,
        max_value_wrong: float = -0.5,
        min_value_correct: float = 0.5,
        max_value_correct: float = 1.0,
        max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            is_correct = accuracy_reward_function(model_output=content, label=sol)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
                cosine = -cosine
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_cosine_scaled_reward(
        min_value_wrong: float = -1.0,
        max_value_wrong: float = -0.5,
        min_value_correct: float = 0.5,
        max_value_correct: float = 1.0,
        max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def extract_code(completion: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def code_reward(completions, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    rewards = []
    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    try:
        """Returns a reward function that evaluates code snippets in a sandbox."""
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
        verification_info = kwargs["verification_info"]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"]))
            )
            for code, info in zip(code_snippets, verification_info)
        ]
        with Sandbox(timeout=30, request_timeout=3) as sbx:
            for script in scripts:
                execution = sbx.run_code(script, language=verification_info["language"])
                try:
                    output = float(execution.text)
                except (TypeError, ValueError):
                    output = 0.0
                rewards.append(output)
    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)
    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward
