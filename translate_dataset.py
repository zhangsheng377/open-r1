from rich import print
import rich.traceback
from tqdm import tqdm

# 启用彩色堆栈跟踪
rich.traceback.install()

from datasets import load_dataset

from utils.chatbot.chatbot import ChatBot

dataset = load_dataset(
    "parquet",
    data_dir="data/OpenThoughts-114k/data",
    data_files="*.parquet",
    split='train'
)
dataset = dataset.take(1000)
print(dataset)

CHINESE_SYSTEM_PROMPT = """你的角色是通过系统的长时间思考过程彻底探索问题，在提供最终精确准确的解决方案之前。这需要参与全面的分析、总结、探索、重新评估、反思、回溯和迭代的周期，以发展经过深思熟虑的思维过程。请将你的回答分为两个主要部
分：Thought和Solution。在Thought部分，使用指定格式详细描述你的推理过程：<|begin_of_thought|> {用'\n\n'分隔的步骤} <|end_of_thought|>
每个步骤应包括详细的考虑，如分析问题、总结相关发现、头脑风暴新想法、验证当前步骤的准确性、完善任何错误以及重新审视之前的步骤。在Solution部分，根据Thought部分的各种尝试、探索和反思，系统地呈现你认为正确的最
终解决方案。该解决方案应保持逻辑性、准确性和简洁性的表达风格，并详细说明达到结论所需的必要步骤，格式如下：<|begin_of_solution|> {最终的格式化、精确且清晰的解决方案} <|end_of_solution|>
现在，请按照上述指南尝试解决以下问题："""

PROMPT = """你是一个大模型语料的翻译员，负责把英文语料翻译成中文语料。需要注意的是，语料中的特殊token不要翻译，例如：<|begin_of_thought|>、<|end_of_thought|>、<|begin_of_solution|>、<|begin_of_solution|>等。在除了思考阶段的别的部分，请只输出译文，其他的话不要输出。




以下是你需要翻译的语料：
{doc}




请输出译文：
"""

chatbot = ChatBot(model_name="deepseek-r1:32b")


# 定义一个函数来处理列 a 并生成新列 b
def process(example):
    example['chinese_system'] = CHINESE_SYSTEM_PROMPT
    example['chinese_conversations'] = []
    for conv in example['conversations']:
        message = [{"role": "user", "content": PROMPT.format(doc=conv['value'])}]
        result = chatbot.chat(messages=message)
        example['chinese_conversations'].append({"from": conv['from'], "value": result})
    return example


with tqdm(total=len(dataset), desc="Processing data") as pbar:
    # 使用 map 方法添加新列
    dataset = dataset.map(process)
    pbar.update(len(dataset))

dataset.to_json("data/OpenThoughts-114k/chinese_data_1000.jsonl", lines=True, force_ascii=False)
