from dataset.load_file_data import get_dataset
import pandas as pd

def count_label(labels, name_labels):
    value_labels = list(set(labels))
    
    import collections
    count_labels = collections.Counter(labels)

    observe_data = pd.DataFrame({f'{name_labels}': value_labels, 'num_labels': [count_labels[value] for value in value_labels]})
    return observe_data

def test_load_data(name = 'test'):
    sentiments, sents, topics = get_dataset(type_data = name)
    
    print(f'Number samples: {len(sents)}')
    
    print('Labels sentiments:')
    print(count_label(labels = sentiments, name_labels = 'sentiments'))
    print('Labels topics:')
    print(count_label(labels = topics, name_labels = 'topics'))


if __name__ == '__main__':
    test_load_data('test')
    

