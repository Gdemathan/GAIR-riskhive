import json


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



