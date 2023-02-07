from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import json
import re

english_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
                     'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                     "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                     "won't", 'wouldn', "wouldn't"]
                     
def cleaning(src):
    string = src
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()
    return string

class Preprocessor():
    def __init__(self, raw_dir, hie_file, save_dir) -> None:
        self.raw_dir = raw_dir
        self.save_dir = save_dir
        self.hie_file = hie_file
        self.datasets = {"train":[], "test":[], "dev":[]}
        self.parent_child_map = {} 
        self.id_label_map = {"root":"root"} 
        self.target_labels = [] 
        self.task = None

    def data_loading(self):
        pass
        
    def build_hierarchy(self):
        pass

    def save(self):
        save_dir = Path(self.save_dir)
        for split, fs in self.datasets.items():
            save_file = (save_dir/"".join([split,".jsonl"]))
            writer = save_file.open("w")
            for f in fs:
                writer.write(json.dumps(f)+"\n")
            writer.flush()
            writer.close()
            print(f"Save preprocessed {self.task} {split} dataset at {str(save_file)}.")
            print(f"\tTotal instance : {len(fs)}")

        save_file = save_dir/"parent_child_map.txt"
        writer = save_file.open("w")
        for k, v in self.parent_child_map.items():
            writer.write(f"{v}\t{k}\n")
        writer.flush()
        writer.close()

        save_file = save_dir/"labels.txt"
        writer = save_file.open("w")
        for k, v in self.id_label_map.items():
            writer.write(f"{k}\t{v}\n")
        writer.flush()
        writer.close()

        save_file = save_dir/"target_labels.txt"
        writer = save_file.open("w")
        for l in self.target_labels:
            writer.write(f"{l}\n")
        writer.flush()
        writer.close()

    def preprocessing(self):
        self.data_loading()
        self.build_hierarchy()
        self.save()


class EURLEX57KPreprocessor(Preprocessor):
    def __init__(self, raw_dir, hie_file, save_dir) -> None:
        super().__init__(raw_dir, hie_file, save_dir)
        self.task = "EURLEX57K"

    def data_loading(self):
        files = {k:list((Path(self.raw_dir)/k).iterdir()) for k in self.datasets.keys()}
        for split, fs in files.items():
            for f in tqdm(fs, desc=f"{self.task}-{split}"):
                article = json.load(f.open())
                did = article["celex_id"]
                labels = article["concepts"]
                text = " ".join([article["title"], article["header"], article["recitals"]," ".join(article["main_body"])])
                token = self.tokenizing(text)
                instance = {"did":did, "text":text, "labels":labels, "token":token}
                self.datasets[split].append(instance)

    def tokenizing(self, text):
        text = cleaning(text)
        token = [word.lower() for word in text.split() if word not in english_stopwords and len(word) > 1]
        return token

    def build_hierarchy(self):
        label_cnt = Counter()
        for dataset in self.datasets.values():
            for instance in dataset:
                label_cnt.update(instance["labels"])
        hierarchy = json.load(Path(self.hie_file).open())

        print(f"Original EURLEX #Label : {len(hierarchy.keys())}")
        print(f"Target EURLEX57K #Label : {len(label_cnt)}")
        print(f"Label Coverage : {len(label_cnt)/len(hierarchy.keys()):0.2f}")
        self.target_labels = list(label_cnt.keys())

        self.id_label_map["-1"] = "nation_virtual"
        self.parent_child_map["-1"]="root"
        for k, v in hierarchy.items():
            if k not in label_cnt:
                continue
            if len(v["parents"]) > 1:
                self.parent_child_map[k] = "-1"
            elif len(v["parents"]) == 0:
                self.parent_child_map[k] = "root"
            else:
                self.parent_child_map[k] = v["parents"][0]
            self.id_label_map[k] = v["label"]

        for _ in range(5):
            childs, parents = list(self.parent_child_map.keys()), list(self.parent_child_map.values())
            for parent in parents:
                if parent not in childs and parent not in ["root", "-1"]:
                    tmp = hierarchy[parent]
                    if len(tmp["parents"]) > 1:
                        self.parent_child_map[parent] = "-1"
                    elif len(tmp["parents"]) == 0:
                        self.parent_child_map[parent] = "root"
                    else:
                        self.parent_child_map[parent] = tmp["parents"][0]
                    self.id_label_map[parent] = tmp["label"]


preprocessors = {"EURLEX57K":EURLEX57KPreprocessor}

def main():
    parser = ArgumentParser()
    # general args
    parser.add_argument("--name", type=str, default=None, required=True,
                        help="Dataset name")
    parser.add_argument("--raw_dir", type=str, default=None, required=True,
                        help="Dataset name")
    parser.add_argument("--hierarchy_file", type=str, default=None, required=True,
                        help="Dataset name")
    parser.add_argument("--save_dir", type=str, default=None, required=True,
                        help="Dataset name")
    
    args = parser.parse_args()

    preprocessor = preprocessors[args.name](args.raw_dir, args.hierarchy_file, args.save_dir)

    preprocessor.preprocessing()

if __name__ == "__main__":
    main()

# python src/preprocessing.py --name=EURLEX57K --raw_dir=[raw_data_dir] --save_dir=[save_dir]  --hierarchy_file=[EURLEX57K.json_file path]