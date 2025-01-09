import json
import logging
import os


class Logger:
    """
    Logger class. It is used to log information and errors.
    This class is used to keep track of the useful logs. When debugging, one can type "print".
    The logs printed by the logger are useful data that can be used to understand the behavior of the program,
    and ought to be kept.

    Args:
        log_file (str): The path to the log file. If None, the logs will not be saved to a file.
    """

    logger = logging.getLogger("masterclass")

    def __init__(self, log_file: str = None):
        self.log_file = log_file
        if self.log_file is not None:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            with open(self.log_file, "w") as f:
                f.write("logger initialized\n")
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="[INFO]: %(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def info(self, *args):
        if self.log_file:
            print(*args)
        string_args = " ".join(map(str, args))
        self.logger.info(string_args)

    def error(self, *args):
        if self.log_file:
            print(*args)
        string_args = " ".join(map(str, args))
        self.logger.error(string_args)


logger = Logger("generated/logs.txt")


def save_json(list_or_dict, fname: str) -> None:
    if len(os.path.dirname(fname)) > 0:
        os.makedirs(os.path.dirname(fname), exist_ok=True)

    with open(fname, "w", encoding="utf-8") as fichier:
        json.dump(list_or_dict, fichier, ensure_ascii=False, indent=4)


def read_json(fname):
    with open(fname, "r", encoding="utf-8") as fichier:
        loaded = json.load(fichier)

    print(f"... File loaded {loaded}")
    return loaded


RISKHIVE_STRING = """
===================================
||                               ||
||          _________            ||
||         |         |           ||
||         |R I S K  |           ||
||         |  H I V E|           ||
||         |_________|           ||
||                               ||
===================================
[STATUS]: All systems go. Ready to analyze some risk!

"""


def init_string():
    logger.info(RISKHIVE_STRING)
