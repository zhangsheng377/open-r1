from time import sleep
from typing import Union, List, Dict

from openai import OpenAI, NOT_GIVEN

OPENAI_BASE_URL = 'http://localhost:11434/v1/'
MODEL_NAME = 'llama-3-chinese-8b-instruct-v3-f16'


# MODEL_NAME = 'qwen2:7b'

class ChatBot:
    def __init__(self, model_name=None):
        self.client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key='ollama',  # required but ignored
        )
        self.model_name = model_name

    def _run_conversation(self, messages: Union[List[Dict[str, str]], str], temperature, tools, stream, stop):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        response = self.client.chat.completions.create(
            messages=messages,
            temperature=temperature,
            stream=stream,
            model=MODEL_NAME if not self.model_name else self.model_name,
            stop=stop,
            tools=tools,
        )
        return response

    def _chat(self, messages: Union[List[Dict[str, str]], str], temperature, tools, stop):
        response = self._run_conversation(messages=messages, temperature=temperature, stream=False, tools=tools,
                                          stop=stop)
        return response.choices[0].message.content

    def _stream_chat(self, messages: Union[List[Dict[str, str]], str], temperature, tools, stop):
        response = self._run_conversation(messages=messages, temperature=temperature, stream=True, tools=tools,
                                          stop=stop)
        for token in response:
            if token.choices[0].finish_reason is not None:
                continue
            yield token.choices[0].delta.content

    def chat(self, messages: Union[List[Dict[str, str]], str], temperature=0.6, tools=NOT_GIVEN, stop=NOT_GIVEN,
             stream=False):
        if not stream:
            return self._chat(messages=messages, temperature=temperature, tools=tools, stop=stop)
        else:
            return self._stream_chat(messages=messages, temperature=temperature, tools=tools, stop=stop)


if __name__ == '__main__':
    chatbot = ChatBot()
    message = [{"role": "user", "content": "hello."}]
    print(chatbot.chat(messages=message))

    print("\n\n\n")

    message = "hello."
    for token in chatbot.chat(messages=message, stream=True):
        print(token, end='', flush=True)
        sleep(0.1)
    print('\n')

    tools = [{
        'type': 'function',
        'function': {
            'name': 'web_search',
            'description': 'Useful for when you need to answer questions about current events.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Tsearch query of web',
                    },
                },
                'required': ['query'],
            },
        },
    },
    ]
    message = [{'role': 'user',
                'content': '你是一个在线个人助手。当被问及你所不知道的问题时，可以调用工具从网上搜索相关信息以回答问题。问题:金陵中学所在的行政区有多少人?'}]
    print(chatbot.chat(messages=message, tools=tools, stream=False))
    print('\n')
    # 传递tools的方法好像没有用。最好也是最清晰的方法就是我自己组装tools的prompt。