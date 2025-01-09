import subprocess
import os


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
    if os.getlogin() == "Luc":
        python_executable = r"C:\Users\Luc\anaconda3\python.exe"

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


PYTHON_TOOL_DICT = {
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


if __name__ == "__main__":
    script = """
import numpy as np
a = np.random.randint(0,100)
print(a)
#1/0
"""
    res = execute_python_script(script)
    print(f"Result of the script: {res}")
