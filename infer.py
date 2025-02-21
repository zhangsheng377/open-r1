import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from peft import get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.chatbot.chatbot_local import ChatBot

# MODEL_PATH = '/mnt/nfs/zsd_server/models/huggingface/Qwen2.5-7B'
MODEL_PATH = '/mnt/nfs/zsd_server/models/huggingface/Qwen2.5-7B-Instruct'
LORA_PATH = '/mnt/nfs/zsd_server/codes/open-r1/output/Qwen2.5-7B-Open-R1-GRPO'

query = r'''What is the coefficient of $x^2y^6$ in the expansion of $\left(\frac{3}{5}x-\frac{y}{2}\right)^8$? Express your answer as a common fraction.'''
query=r"""你是一个大模型回答标注员，你需要根据[参考答案]去给大模型的[模型回答]打分，打分范围为0-10，只需输出数字即可，不要有其他任何文字。




<参考答案>
To solve this problem, we need to analyze the statements made by the inhabitants of the island. Each person claims that the next 10 people in the clockwise direction are liars.

Let's denote the inhabitants as \(1, 2, \ldots, 99\) standing in a clockwise circle. Each person \(i\) makes the statement about the people \(i+1, i+2, \ldots, i+10\).

We can identify the following when solving the problem:

1. If person \(i\) is a knight (truth-teller), then all the people \(i+1\) to \(i+10\) must be liars.
2. If person \(i\) is a liar, then at least one person among \(i+1\) to \(i+10\) must be a knight because the statement is false.

We can set up a system of equations to solve the problem, but we also need to consider the modulo operation to wrap around the circle.

Let's analyze the problem step-by-step to see if we can identify patterns or directly count the possibilities.

First, let's set up the problem in Python using sympy to solve it:

```python
import sympy as sp

# Define the number of inhabitants
n = 99

# Create a list to represent whether each person is a knight (True) or liar (False)
people = sp.symbols('p0:%d' % n)

# Create a list to store the equations
equations = []

# Generate the equations based on the statements
for i in range(n):
    # The next 10 people in the clockwise direction
    next_10 = [(i + j) % n for j in range(1, 11)]
    # If person i is a knight (truth-teller)
    truth_statement = sp.And(*[~people[j] for j in next_10])
    # If person i is a liar (liar)
    lie_statement = sp.Or(*[people[j] for j in next_10])
    # Person i's statement
    equations.append(sp.Implies(people[i], truth_statement))
    equations.append(sp.Implies(~people[i], lie_statement))

# Solve the system of equations for the number of true values
solution = sp.satisfiable(sp.And(*equations), all_models=True)
knights_count = []

# Iterate through all possible solutions and count the knights
for sol in solution:
    knights_count.append(sum([sol[person] for person in people]))

# Find the unique possible counts of knights
unique_knights_count = set(knights_count)
print(unique_knights_count)
```
```output
{9}
```
The output indicates that the unique possible count of knights among the 99 inhabitants who stood in the circle is 9. Therefore, there could be \(\boxed{9}\) knights on the island.
</参考答案>




<模型回答>
Let's analyze the situation:

If a knight says "All ten people following me in a clockwise direction are liars," it means there can't be any other knights among those ten people. If there were a knight among those ten, the speaker would be lying, which contradicts their nature as truth-tellers.

On the other hand, if a liar says the same, it means there must be at least one knight among those ten people. If all ten were liars, the speaker would be telling the truth, which contradicts their nature as liars.

So, we have two scenarios for each person:
1. They are a knight, and there are no knights among the next ten people (clockwise).
2. They are a liar, and there is at least one knight among the next ten people.

Now let's consider the implications:

In scenario 1, if a person is a knight, the next nine people must be liars, and the tenth person can be either a knight or a liar. If the tenth person is a knight, we'll call them "Person A." Since there cannot be any knights among the next ten people for Person A, they must be a liar. But this creates a contradiction: Person A was initially assumed to be a knight, yet we just concluded they have to be a liar. Therefore, Person A cannot be a knight, meaning they must be a liar. This pattern continues indefinitely throughout the circle.

However, in scenario 2, if a person is a liar, it implies that there is at least one knight among the next ten people. If all ten were liars, then the person saying so would be telling the truth, which contradicts their nature as liars. Thus, for every liar in the circle, there must be at least one knight following them.

Considering both scenarios, the only way to satisfy the conditions is if there are exactly nine knights and one liar in the circle, alternating like a checkerboard pattern. The liar is surrounded by knights, while the knights are surrounded by liars, satisfying the conditions outlined by everyone's statements.

Therefore, based on our analysis, there could be $\boxed{9}$ knights among those who stood in the circle.
</模型回答>




你只需要给出0-10的数字即可，不要有其他多余描述。"""

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    # torch_dtype=torch.float16,
    # device_map="cpu",
)
# ).to("cpu")
# model = PeftModel.from_pretrained(model, LORA_PATH, device_map="cpu")

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
