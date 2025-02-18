import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    # torch_dtype=torch.float16,
    device_map="cpu",
).to("cpu")
model = PeftModel.from_pretrained(model, LORA_PATH, device_map="cpu")

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

r"""
<think>First, we use the binomial theorem to expand the expression. The binomial theorem states that $(a+b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k$. In this case, $a = \frac{3}{5}x$, $b = -\frac{y}{2}$, and $n = 8$. We want to find the coefficient of $x^2y^6$, so we need to find the term where the exponents of $x$ and $y$ add up to 8.

Using the binomial theorem, the general term in the expansion is $\binom{8}{k} \left(\frac{3}{5}x\right)^{8-k} \left(-\frac{y}{2}\right)^k$. We want the term where $8-k = 2$ and $k = 6$, so we substitute these values into the general term:

$\binom{8}{6} \left(\frac{3}{5}x\right)^2 \left(-\frac{y}{2}\right)^6 = \binom{8}{6} \left(\frac{9}{25}x^2\right) \left(\frac{1}{64}y^6\right)$.

Simplifying this expression, we get $\frac{8!}{6!2!} \cdot \frac{9}{25} \cdot \frac{1}{64} x^2 y^6 = \frac{28 \cdot 9}{25 \cdot 64} x^2 y^6 = \frac{63}{400} x^2 y^6$.

Therefore, the coefficient of $x^2y^6$ in the expansion is $\frac{63}{400}$.</think>
<answer> $\frac{63}{400}$ </answer>
"""
