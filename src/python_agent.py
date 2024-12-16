import subprocess
import os
from openai import OpenAI
import dotenv
from utils import logger, save_json
import json

dotenv.load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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
        print(f"Script written to sandbox_script.py: \n{script}")

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
        self, client: OpenAI, max_retries: int = 3, tool_prompt: str = DEFAULT_TOOL_PROMPT
    ):
        self.client = client
        self.num_retries = 0
        self.retry_prompt = RETRY_PROMPT
        self.tool_prompt = tool_prompt
        self.max_retries = max_retries

    def _append_tool_prompt_to_system(self, messages: list[dict]) -> list[dict]:
        def append_prompt(message):
            if message["role"] == "system":
                return {
                    **message,
                    "content": f"{message.get('content', '')} \n {self.tool_prompt}",
                }
            return message

        return list(map(append_prompt, messages))

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
        A custom implementation of the completions.create method.
        This wrapper prompts the system to use python if it needs to.
        """
        if self.num_retries >= self.max_retries:
            save_json(messages, "generated/messages.json")
            raise ToolError("Max retries reached. Returning no output.")
        answer = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self._append_tool_prompt_to_system(messages),
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            tools=[tool_dict],
            **kwargs,
        )
        if answer.choices[0].finish_reason != "tool_calls":
            self.num_retries = 0
            all_messages = messages + [
                {"role": "assistant", "content": answer.choices[0].message.content},
            ]
            save_json(all_messages, "messages.json")
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
                self.num_retries += 1
                return self.ask_with_python(
                    messages=messages
                    + [
                        {
                            "role": "assistant",
                            "content": "It appears my script crashed.",
                        },
                        {"role": "user", "content": craft_error_prompt(str(e))},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stream=stream,
                    stop=stop,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    user=user,
                    **kwargs,
                )
            except Exception as e:
                logger.error("An unexpected error occurred: ", e)
                raise ToolError("An unexpected error occurred.")
            if result is None:
                if self.num_retries >= self.max_retries:
                    logger.info("Max retries reached. Returning no output.")
                    raise ToolError("Max retries reached. Returning no output.")
                logger.info("The script returned no output. Trying again")
                self.num_retries += 1
                return self.ask_with_python(
                    messages=messages
                    + [
                        {
                            "role": "assistant",
                            "content": "Script executed : \n" + script,
                        },
                        {"role": "user", "content": self.retry_prompt},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stream=stream,
                    stop=stop,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    user=user,
                    **kwargs,
                )
            logger.info("Script executed successfully.")
            logger.info(f"Prompting with the result: {result}")
            self.num_retries += 1
            return self.ask_with_python(
                messages=messages
                + [
                    {
                        "role": "assistant",
                        "content": f"My python script returned {result}",
                    },
                    {"role": "user", "content": craft_result_message(result)},
                ],
                temperature=temperature,
                top_p=top_p,
                n=n,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                user=user,
                **kwargs,
            )
            


if __name__ == "__main__":
    agent = PythonAgent(client)
    answer = agent.ask_with_python(
        [
            {"role": "system", "content": "Hello!"},
            {"role": "user", "content": USER_PROMPT},
        ]
    )
    print("answer: ", answer.choices[0].message.content)
