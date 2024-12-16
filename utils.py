import json
import logging


class Logger:
    logger = logging.getLogger("masterclass")

    def __init__(self, log_file: str = None):
        self.log_file = log_file
        if self.log_file is not None:
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


logger = Logger()


def save_json(list_or_dict,fname:str)->None:
    with open(fname, "w", encoding="utf-8") as fichier:
        json.dump(list_or_dict, fichier, ensure_ascii=False, indent=4)
    
    print(f'... File saved {fname}')


def read_json(fname):
    with open(fname, "r", encoding="utf-8") as fichier:
        loaded = json.load(fichier)

    print(f'... File loaded {loaded}')
    return loaded



if __name__=='__main__':
    ma_liste = ["élément 1", "élément 2", "élément 3"]
    save_json(ma_liste,'RAG.json')

    read_json('RAG.json')



