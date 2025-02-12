import torch
from peft import get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.chatbot.chatbot_local import ChatBot

MODEL_PATH = '/mnt/nfs/zsd_server/models/huggingface/Qwen2.5-7B'
LORA_PATH = '/mnt/nfs/zsd_server/codes/open-r1/output/Qwen2.5-7B-Open-R1-GRPO'

query = r'''What is the coefficient of $x^2y^6$ in the expansion of $\left(\frac{3}{5}x-\frac{y}{2}\right)^8$? Express your answer as a common fraction.'''

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, LORA_PATH)

chatbot = ChatBot(model=model, tokenizer=tokenizer)

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
# message = f"""You will be given a problem. Please reason step by step, and put your final answer within \\boxed{{}}:\n{query}"""
message = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": query},
]
for token in chatbot.chat(messages=message, stream=True, skip_special_tokens=False):
    print(token, end='', flush=True)
    # sleep(0.1)
print('\n')
