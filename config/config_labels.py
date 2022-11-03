from email.policy import default
import os
from dataset.load_file_data import get_dataset
import json
from config import default_path


def get_label(flag = "topic"):
    sentiments, sents, topics = get_dataset(type_data = 'train')
    sentiments = [int(sentiment) for sentiment in sentiments]
    topics = [int(topic) for topic in topics]

    if flag == "topic":
        return set(topics)
    elif flag == "sentiment":
        return set(sentiments)

def save_json(dic, path):
    f_json = json.dumps(dic)
    f = open(path,"w")
    f.write(f_json)
    f.close()

def read_json(path):
    with open(path, 'r') as f:
        data = f.read()
    js = json.loads(data)
    return js

if os.path.exists(path = default_path.path_label):
    label2index = read_json(default_path.path_label)
else:
    label2index = {k:v for k, v in enumerate(get_label(default_path.type_label))}
    save_json(label2index, path = default_path.path_label)




