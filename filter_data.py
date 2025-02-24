import os.path

from rich import print
import rich.traceback
from datasets import load_dataset
from transformers import AutoTokenizer

# 启用彩色堆栈跟踪
rich.traceback.install()

TOKENIZER_PATH = "/mnt/nfs/zsd_server/models/huggingface/Qwen2.5-7B-Instruct"
DATASET_PATH = "/mnt/nfs/zsd_server/codes/open-r1/data/Chinese-DeepSeek-R1-Distill-data-110k"
FILTER_DATASET_PATH = "/mnt/nfs/zsd_server/codes/open-r1/data/Chinese-DeepSeek-R1-Distill-data-110k_filter"
MAX_LENGTH = 640

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

dataset = load_dataset(
    "json",
    data_dir=DATASET_PATH,
    data_files="*.jsonl",
)
print(dataset)


# 定义过滤函数
def filter_by_length(example):
    return example["prompt_tokens_len"] + example["content_tokens_len"] < MAX_LENGTH


# 应用过滤
filtered_dataset = dataset.filter(filter_by_length)


def process(example):
    return {"problem": example["input"], "solution": example["content"]}


original_columns = filtered_dataset["train"].column_names
filtered_dataset = filtered_dataset.map(process, remove_columns=original_columns)

# 打印过滤前后的样本数量
print(f"Original dataset size: {len(dataset['train'])}")
print(f"Filtered dataset size: {len(filtered_dataset['train'])}")

# （可选）保存过滤后的数据集
# filtered_dataset.save_to_disk(FILTER_DATASET_PATH)
filtered_dataset["train"].to_json(os.path.join(FILTER_DATASET_PATH, "data", "train.jsonl"), lines=True,
                                  force_ascii=False)
