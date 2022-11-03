def get_data(path):
    with open(file = path, mode = 'r', encoding = 'utf8') as f:
        data = [line.strip() for line in f.readlines()]
    return data

def get_dataset(type_data):
    path_sentiments = f'dataset/_UIT-VSFC/{type_data}/sentiments.txt'
    path_sents = f'dataset/_UIT-VSFC/{type_data}/sents.txt'
    path_topics = f'dataset/_UIT-VSFC/{type_data}/topics.txt'
    
    sentiments = get_data(path_sentiments)
    sents = get_data(path_sents)
    topics = get_data(path_topics)
    return sentiments, sents, topics