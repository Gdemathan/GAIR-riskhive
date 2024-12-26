
if __name__=='__main__':
    from handler_rag import RAG
    from handler_python import GPT_tool_python_script,execute_python_script,PythonError
    from client import openai_client
    from utils import save_json
else:
    from src.handler_rag import RAG
    from src.handler_python import GPT_tool_python_script,execute_python_script,PythonError
    from src.client import openai_client
    from src.utils import save_json

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from dataclasses import dataclass
import json
import re

@dataclass
class API_params:
    temperature: float = None,
    top_p: float = None,
    n: int = 1,
    stream: bool = False,
    stop: str = None,
    max_tokens: int = None,
    presence_penalty: float = None,
    frequency_penalty: float = None,
    logit_bias: dict = None,
    user: str = None,
    model: str ="gpt-4o-mini",

    def to_dict(self):
        dico = self.__dict__
        return {k:v[0] if isinstance(v,tuple) else v for k,v in dico.items()}



class _RagPythonAgent:

    def __init__(
            self,
            system_prompt:str,
            question:str,
            rag:RAG,
            tools:list[dict],
            logs_path:str = 'generated/logs.json',
            client:OpenAI = openai_client,
            api_params:API_params = API_params(),
            max_retries: int = 4,
    ):
        self.rag = rag
        self.logs:list[dict] = [
            {'role':'system', 'content':system_prompt},
            {'role':'user', 'content':self._improve_question(question)}
        ]

        self.tools = tools if isinstance(tools,list) else [tools]
        self.logs_path = logs_path
        self.client = client
        self.max_retries = max_retries
        self.api_params = api_params.to_dict()

    def _improve_question(self,q):
        """To be improved in the real Agent."""
        return q

    def _ask_llm(self, messages:list[dict], tools:list=[], api_params:dict={})->ChatCompletion:
        return self.client.chat.completions.create(
            messages=messages, tools=tools, **api_params
        )
    
    def _parse_answer(self,chat_completion:ChatCompletion)->bool:

        finish_reason = chat_completion.choices[0].finish_reason
        message = chat_completion.choices[0].message.content


        if finish_reason != "tool_calls":
            self.logs += [{'role':'assistant','content':message}]

            if 'I need python now' in message:
                self.logs += [{'role':'user','content':"""Please define your tool request and I will give you the answer. You should put the output as a print() at the end of the code."""}]
                return False
            else:
                letters = re.findall(r'[a-d]', message)
                self.logs += [{'role':'assistant','content':letters[-1]}]
                return True
        else:
            tool_call = chat_completion.choices[0].message.tool_calls[0]
            
            if tool_call.function.name == "execute_python_script":
                arguments = json.loads(tool_call.function.arguments)
                script = arguments["script"]
                self.logs += [{'role':'assistant','content':f"""Trying to execute the following script: \n\n {script}"""}]
                try:
                    result = execute_python_script(script)
                except PythonError as e:
                    self.logs += [{'role':'user','content':"""Please define a working tool request and I will give you the answer."""}]
                    return False
                except Exception as e:
                    self.logs += [{'role':'user','content':f"""There has been an error with your code: {e}. Please use the tool and define a new tool_call."""}]
                    return False
            
                if result is None:
                    self.logs += [{'role':'user','content':f"""The code should print something at the end. I will return you the answer after. Please try again generating the tool_call."""}]
                    return False
                else:
                    self.logs += [{'role':'user','content':f"""Result from your python request : {result}.
Answer the initial question using this result. If possible do not use python again."""}]
                    return False

    def run(self):
        final_answer = False
        chat_len = 0
        while not final_answer and chat_len<self.max_retries:
            chat_len+=1

            chat = self._ask_llm(self.logs,self.tools,self.api_params)

            final_answer = self._parse_answer(chat)
        
        self.save_logs()
        return self.answer()

    def answer(self):
        return self.logs[-1]['content']

    def save_logs(self):
        save_json(self.logs, self.logs_path)


class RagPythonAgent(_RagPythonAgent):
    def _improve_question(self,q):
        rag_info = self.rag.best_fit(q)
        return q + f"""

[Additional Knowledge]:
{rag_info}
"""


if __name__=='__main__':
    system_prompt = """You are a Realiability Engineer. You are to answer to a Multiple Choice Question.
For each question, there is only 1 possible answer.
Each question is accompagned with some additionnal piece of knowledge you are able to use.

Your answer should always have the following form:

***Explanations:***
Some explanations you give to answer the questions. You must explain all your reasoning steps. You should never do any computations yourself in this part. All the computations will eventually be described but only litterally and done using python in a second time.


***Next Step:***
At this stage, you can only response with one of the following things:
- the letter corresponding to the answer in the question. For example, you answer "a" if the answer is "[a]". Please only give the letter and nother other letter or font.
- if you think that you need to perform computations, you will be able to ask. In this case, at this stage, please write "I need python now".
"""
    question = """[Question]: 1. Battery life has been measured as normally distributed with  mean equal to 150 hours and variance of 400 hours. Find the B10 life. 

[Choices]: [a] 124.4 hrs  | [b] 175.6 hrs"""
    rag = RAG('RAG.json')
    tools = GPT_tool_python_script
    api_params = API_params(temperature=6)


    agent = _RagPythonAgent(system_prompt,question,rag,tools)
    answer = agent.run()
    print(answer)
