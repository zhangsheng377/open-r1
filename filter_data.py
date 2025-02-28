import os.path

from rich import print
import rich.traceback
from datasets import load_dataset
from transformers import AutoTokenizer

# 启用彩色堆栈跟踪
rich.traceback.install()

TOKENIZER_PATH = "/mnt/nfs/zsd_server/models/huggingface/Qwen2.5-7B-Instruct"
DATASET_PATH = "/mnt/nfs/zsd_server/codes/open-r1/data/chinese-sft-stem-zh-hans"
FILTER_DATASET_PATH = "/mnt/nfs/zsd_server/codes/open-r1/data/chinese-sft-stem-zh-hans/filter"
MAX_LENGTH = 640
SYSTEM_PROMPT = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# dataset = load_dataset(
#     "parquet",
#     data_dir=DATASET_PATH,
#     data_files="*.parquet",
# )
dataset = load_dataset(DATASET_PATH)
print(dataset)


# 定义过滤函数
def filter_by_length(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example['conversations'][0]['value']},
        {"role": "assistant", "content": example['conversations'][1]['value']},
    ]
    chat_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt",
                                                return_dict=True)
    return chat_tokens['input_ids'].size(1) < MAX_LENGTH


# 应用过滤
filtered_dataset = dataset.filter(filter_by_length)


def process(example):
    return {"problem": example['conversations'][0]['value'], "solution": example['conversations'][1]['value']}


original_columns = filtered_dataset["train"].column_names
filtered_dataset = filtered_dataset.map(process, remove_columns=original_columns)

# 打印过滤前后的样本数量
print(f"Original dataset size: {len(dataset['train'])}")
print(f"Filtered dataset size: {len(filtered_dataset['train'])}")

# （可选）保存过滤后的数据集
# filtered_dataset.save_to_disk(FILTER_DATASET_PATH)
filtered_dataset["train"].to_json(os.path.join(FILTER_DATASET_PATH, "data", "train.jsonl"), lines=True,
                                  force_ascii=False)
