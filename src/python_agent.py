from openai import OpenAI
import json

if __name__ != "__main__":
    from src.utils import logger, save_json
    from src.client import openai_client
    from src.handler_python import execute_python_script, PythonError, PYTHON_TOOL_DICT


DEFAULT_TOOL_PROMPT = """You can use python whenever you need to perform complex calculations or operations.
 When needed write a python script that prints only the final result.
 Example: 
 
 list = [4,1,2]
 print(sorted(list))
 Do not forget the "print" statement.
 """


RETRY_PROMPT = """Be careful, you script does not return the result.
Make sure it prints the result at the end of the script."""


def craft_error_prompt(message: str):
    return f"""Please make code that works, your script just crashed. Here is the error: {message}.
    Please fix the cause of this error in your script. For instance, np.gamma does not exist, it is np.random.gamma."""


def craft_result_message(result: str):
    return f"""The python script successfully returned: {result}.
    Answer the initial question using this result. If possible do not use python again"""


class ToolError(Exception):
    def __init__(self, message):
        super().__init__(message)


class PythonAgent:
    """
    A class that gives access to python script execution.
    It injects instructions in the system prompt, while giving access to the tool.
    It handle all 3 cases:
    - success
    - the created script crashes
        -> will prompt to re-write the script, with the error
    - the created script returns None
        -> will prompt to return something

    """

    def __init__(
        self,
        client: OpenAI,
        max_retries: int = 3,
        tool_prompt: str = DEFAULT_TOOL_PROMPT,
        messages_log_path: str = "generated/messages.json",
    ):
        self.client = client
        self.client._default_ask_llm = self.client.chat.completions.create
        self._num_retries = 0
        self.retry_prompt = RETRY_PROMPT
        self.tool_prompt = tool_prompt
        self._max_retries = max_retries
        self._api_params = {}
        self._override_args = True
        self._messages_log_path = messages_log_path
        self._messages = []
        logger.info("--- Python agent initialized ---\n")

    def _reset_internals(self):
        self._override_args = True
        self._num_retries = 0
        self._api_params = {}
        self._messages = []

    def _append_tool_prompt_to_system(self, messages: list[dict]) -> list[dict]:
        def append_prompt(message):
            if message["role"] == "system":
                return {
                    **message,
                    "content": f"{message.get('content', '')}\n{self.tool_prompt}",
                }
            return message

        return list(map(append_prompt, messages))

    def _ask_llm(self, messages: list[dict]):
        appended_messages = self._append_tool_prompt_to_system(messages)
        return self.client._default_ask_llm(
            messages=appended_messages, **self._api_params
        )

    def ask_with_python(
        self,
        messages: list,
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
        **kwargs,
    ):
        """
        Ask a question and execute python script if needed. Params are the same as the OpenAI API.
        """
        self._messages = messages
        if self._num_retries >= self._max_retries:
            save_json(self._messages, self._messages_log_path)
            self._reset_internals()
            raise ToolError("Max retries reached. Returning no output.")

        if self._override_args:
            self._messages = self._append_tool_prompt_to_system(self._messages)
            self._api_params = {
                "model": "gpt-4o-mini",
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "stream": stream,
                "stop": stop,
                "max_tokens": max_tokens,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "user": user,
                "tools": [PYTHON_TOOL_DICT],
                **kwargs,
            }
            self._override_args = False

        answer = self._ask_llm(self._messages)

        if answer.choices[0].finish_reason == "stop":
            self._num_retries = 0
            all_messages = self._messages + [
                {"role": "assistant", "content": answer.choices[0].message.content}
            ]
            save_json(all_messages, self._messages_log_path)
            return answer

        tool_call = answer.choices[0].message.tool_calls[0]
        if tool_call.function.name == "execute_python_script":
            logger.info("Found a python script to execute.")
            arguments = json.loads(tool_call.function.arguments)
            script = arguments["script"]
            try:
                logger.info(f"Executing the following script: \n\n{script}\n```")
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": f"Executing script: \n```python\n{script}\n",
                    }
                )
                result = execute_python_script(script)
            except PythonError as e:
                logger.error(f"An error occurred: {e}. Scolding model and retrying...")
                self._num_retries += 1
                return self.ask_with_python(
                    messages=self._messages
                    + [
                        {
                            "role": "assistant",
                            "content": f"Script executed: \n{script}",
                        },
                        {"role": "user", "content": craft_error_prompt(str(e))},
                    ],
                    **self._api_params,
                )
            except Exception as e:
                logger.error("An unexpected error occurred: ", e)
                raise ToolError("An unexpected error occurred.")

            if result is None:
                logger.info("The script returned no output. Trying again")
                self._num_retries += 1
                return self.ask_with_python(
                    messages=self._messages
                    + [
                        {
                            "role": "assistant",
                            "content": "Script executed : \n" + script,
                        },
                        {"role": "user", "content": self.retry_prompt},
                    ],
                    **self._api_params,
                )

            logger.info("Script executed successfully.")
            logger.info(f"Prompting with the result: {result}")
            self._num_retries = 0
            response = self.ask_with_python(
                messages=self._messages
                + [
                    {
                        "role": "assistant",
                        "content": f"My python script returned {result}",
                    },
                    {"role": "user", "content": craft_result_message(result)},
                ],
                **self._api_params,
            )
            self._reset_internals()
            return response

    @classmethod
    def inject_python(
        cls,
        openai_client: OpenAI,
        messages_log_path: str = "generated/messages.json",
        tool_prompt: str = DEFAULT_TOOL_PROMPT,
    ):
        """
        Injects the python agent directly in the OpenAI API.
        """
        agent = cls(
            openai_client,
            max_retries=3,
            messages_log_path=messages_log_path,
            tool_prompt=tool_prompt,
        )
        openai_client.chat.completions.create = agent.ask_with_python
        return openai_client


if __name__ == "__main__":
    from utils import logger, save_json
    from client import openai_client
    from handler_python import execute_python_script, PythonError, PYTHON_TOOL_DICT

    USER_PROMPT = """Please sort this list : [1,6,3,5,2,5,6,2,1]."""

    PythonAgent.inject_python(openai_client)
    answer = openai_client.chat.completions.create(
        [
            {"role": "system", "content": "Hello!"},
            {"role": "user", "content": USER_PROMPT},
        ]
    )
