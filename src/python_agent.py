import subprocess
import os
from openai import OpenAI
import json

if __name__ == "__main__":  # for local file testing purposes
    from utils import logger, save_json
    from client import openai_client
else:
    from src.utils import logger, save_json


class PythonError(Exception):
    def __init__(self, message):
        super().__init__(message)


def execute_python_script(script: str) -> str:
    time_out = 5

    venv_path = ".venv"
    if os.name == "nt":
        python_executable = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(venv_path, "bin", "python")

    if not os.path.exists(python_executable):
        raise FileNotFoundError(f"Python interpreter not found at: {python_executable}")

    with open("sandbox_script.py", "w") as file:
        file.write(script)

    try:
        result = subprocess.run(
            [python_executable, "sandbox_script.py"],
            capture_output=True,
            text=True,
            timeout=time_out,
        )
    finally:
        if os.path.exists("sandbox_script.py"):
            os.remove("sandbox_script.py")

    if result.returncode != 0:
        raise PythonError(f"Script execution failed: {result.stderr.strip()}")

    if result.stdout.strip() == "":
        return None

    return result.stdout.strip()


tool_dict = {
    "type": "function",
    "function": {
        "name": "execute_python_script",
        "description": "Executes a python script in a sandboxed environment. Return the stdout printed by the script.",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": ["script"],
            "properties": {
                "script": {
                    "type": "string",
                    "description": "The python script to execute.",
                }
            },
            "additionalProperties": False,
        },
    },
}

DEFAULT_TOOL_PROMPT = """You can use python whenever you need to perform complex calculations or operations.
 When needed write a python script that returns the result."""

USER_PROMPT = """Please sort this list : [1,6,3,5,2,5,6,2,1]."""

USER_EMPTY_PROMPT = """Please sort this list : [1,6,3,5,2,5,6,2,1].
For debug purpose, if you use python please start by making it bug : it must not return anything"""

USER_BUGGED_PROMPT = """Please sort this list : [1,6,3,5,2,5,6,2,1].
For debug purpose, if you use python please start by making it bug. 
You must introduce this line '1/0', the script has to crash (not only return nothing)"""

RETRY_PROMPT = """I'm sorry, but your script did not return shit. Please make a script that actually returns the result"""


def craft_error_prompt(message: str):
    return f"""Please make code that works, your script just crashed mate. Here is the error: {message}"""


def craft_result_message(result: str):
    return f"""The python script successfully returned: {result}.
    Answer the initial question using this result. If possible do not use python again"""


class ToolError(Exception):
    def __init__(self, message):
        super().__init__(message)


class PythonAgent:
    def __init__(
        self,
        client: OpenAI,
        max_retries: int = 3,
        tool_prompt: str = DEFAULT_TOOL_PROMPT,
        messages_log_path: str = "generated/messages.json",
    ):
        self.client = client
        self._num_retries = 0
        self.retry_prompt = RETRY_PROMPT
        self.tool_prompt = tool_prompt
        self._max_retries = max_retries
        self._api_params = {}
        self._override_args = True
        self._messages_log_path = messages_log_path

    def _reset_internals(self):
        self._override_args = True
        self._num_retries = 0
        self._api_params = {}

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
        return self.client.chat.completions.create(
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
        messages_copy = messages.copy()
        if self._num_retries >= self._max_retries:
            save_json(messages_copy, self._messages_log_path)
            self._reset_internals()
            raise ToolError("Max retries reached. Returning no output.")

        if self._override_args:
            messages_copy = self._append_tool_prompt_to_system(messages_copy)
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
                "tools": [tool_dict],
                **kwargs,
            }

        answer = self._ask_llm(messages_copy)

        if answer.choices[0].finish_reason != "tool_calls":
            self._num_retries = 0
            all_messages = messages_copy + [
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
                logger.info("Executing script...")
                result = execute_python_script(script)
            except PythonError as e:
                logger.error(f"An error occurred: {e}. Scolding model and retrying...")
                self._num_retries += 1
                return self.ask_with_python(
                    messages=messages_copy
                    + [
                        {
                            "role": "assistant",
                            "content": "It appears my script crashed.",
                        },
                        {"role": "user", "content": craft_error_prompt(str(e))},
                    ],
                    **self._api_params,
                )
            except Exception as e:
                logger.error("An unexpected error occurred: ", e)
                raise ToolError("An unexpected error occurred.")

            if result is None:
                if self._num_retries >= self._max_retries:
                    logger.info("Max retries reached. Returning no output.")
                    raise ToolError("Max retries reached. Returning no output.")
                logger.info("The script returned no output. Trying again")
                self._num_retries += 1
                return self.ask_with_python(
                    messages=messages_copy
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
            self._num_retries += 1
            return self.ask_with_python(
                messages=messages_copy
                + [
                    {
                        "role": "assistant",
                        "content": f"My python script returned {result}",
                    },
                    {"role": "user", "content": craft_result_message(result)},
                ],
                **self._api_params,
            )


if __name__ == "__main__":
    success_agent = PythonAgent(openai_client)
    answer = success_agent.ask_with_python(
        [
            {"role": "system", "content": "Hello!"},
            {"role": "user", "content": USER_PROMPT},
        ]
    )
    empty_agent = PythonAgent(
        openai_client, messages_log_path="generated/empty_messages.json"
    )
    answer = empty_agent.ask_with_python(
        [
            {"role": "system", "content": "Hello!"},
            {"role": "user", "content": USER_EMPTY_PROMPT},
        ]
    )
    bug_agent = PythonAgent(
        openai_client, messages_log_path="generated/bug_messages.json"
    )
    answer = bug_agent.ask_with_python(
        [
            {"role": "system", "content": "Hello!"},
            {"role": "user", "content": USER_BUGGED_PROMPT},
        ]
    )
