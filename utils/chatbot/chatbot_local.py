from threading import Thread
from time import sleep
from typing import Union, List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria, \
    StoppingCriteriaList


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords: list, tokenizer, device="cpu"):
        self.keywords_ids = [
            tokenizer.encode(w, add_special_tokens=False, return_tensors='pt').squeeze(0).to(device)
            for w in keywords]
        self.keywords_ids = [ids if ids.dim() == 0 else ids for ids in self.keywords_ids]
        print(f"stop_keywords_ids : {self.keywords_ids}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for keyword in self.keywords_ids:
            if keyword.dim() == 0:
                keyword_len = 1
            else:
                keyword_len = len(keyword)
            if len(input_ids[0]) < keyword_len:
                return False
            if input_ids[0][-keyword_len:].equal(keyword):
                return True
            else:
                return False


class ChatBot:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model

    def _run_conversation(self, messages: Union[List[Dict[str, str]], str], temperature, stream, stop):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        chat_tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                         return_dict=True).to(self.model.device)

        streamer = None
        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, timeout=600.0, skip_prompt=True, skip_special_tokens=True)

        stopping_criteria = None
        if stop:
            stop_criteria = KeywordsStoppingCriteria(keywords=stop, tokenizer=tokenizer, device=self.model.device)
            stopping_criteria = StoppingCriteriaList([stop_criteria])

        do_sample = True
        top_p = 0.9
        if not temperature:
            temperature = None
            do_sample = False
            top_p = None

        generate_kwargs = dict(
            input_ids=chat_tokens["input_ids"],
            attention_mask=chat_tokens["attention_mask"],
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if streamer:
            with torch.no_grad():
                thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
                thread.start()
            for new_text in streamer:
                yield new_text
        else:
            outputs = self.model.generate(**generate_kwargs)
            outputs_token = []
            for chat_token, output in zip(chat_tokens["input_ids"], outputs):
                outputs_token.append(output[len(chat_token):])
            outputs = self.tokenizer.batch_decode(outputs_token, skip_prompt=True, skip_special_tokens=True)
            yield outputs[0]

    def _chat(self, messages: Union[List[Dict[str, str]], str], temperature, stop):
        result = ""
        for token in self._run_conversation(messages=messages, temperature=temperature, stream=False, stop=stop):
            result += token
        return result

    def _stream_chat(self, messages: Union[List[Dict[str, str]], str], temperature, stop):
        response = self._run_conversation(messages=messages, temperature=temperature, stream=True, stop=stop)
        for token in response:
            yield token

    def chat(self, messages: Union[List[Dict[str, str]], str], temperature=0.6, stop=None, stream=False):
        if not stream:
            return self._chat(messages=messages, temperature=temperature, stop=stop)
        else:
            return self._stream_chat(messages=messages, temperature=temperature, stop=stop)


if __name__ == "__main__":
    from peft import get_peft_model, PeftModel
    MODEL_PATH = '/mnt/nfs/zsd_server/models/huggingface/Qwen2.5-7B'
    LORA_PATH = '/mnt/nfs/zsd_server/codes/open-r1/output/Qwen2.5-7B-Open-R1-GRPO'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, LORA_PATH)

    chatbot = ChatBot(model=model, tokenizer=tokenizer)

    message = "hello."
    for token in chatbot.chat(messages=message, stream=True):
        print(token, end='', flush=True)
        # sleep(0.1)
    print('\n')