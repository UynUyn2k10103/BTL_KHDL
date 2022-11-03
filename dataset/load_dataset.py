from torch.utils.data import Dataset
from dataset.load_file_data import get_dataset
import numpy as np
import torch
from config.default_path import flag

def keep(items):
    return items

def np_to_tensor(items):
    tensors = [torch.from_numpy(item) for item in items]
    tensors = torch.stack(tensors,dim = 0)
    return tensors

global TENSOR_TYPES 
TENSOR_TYPES = {
    'sent': keep,
    'sentiment': torch.LongTensor,
    'topic': torch.LongTensor,
    'attention_mask': torch.LongTensor,
    'transform_matrix': np_to_tensor,
    'indices': torch.LongTensor,
    'bert_length': torch.LongTensor,
    'word_length': torch.LongTensor,
    'target': torch.LongTensor
}

class VSFC_DataSet(Dataset):
    def __init__ (self, name, tokenizer, args, flag = True):
        if flag:
            sentiments, sents, topics = get_dataset(type_data = name)
            self.sents = sents
            self.sentiments = [int(sentiment) for sentiment in sentiments]
            self.topics = [int(topic) for topic in topics]

            assert len(sents) == len(sentiments)
            assert len(sents) == len(topics)
        else:
            self.sents = [name]
            self.sentiments = [0]
            self.topics = [0]
        
        self.type_label = args.type_label
        self.tokenizer = tokenizer
        self.label2index = args.label2index
            
        self.CLS = self.tokenizer.cls_token_id
        self.PAD = self.tokenizer.pad_token_id
        self.SEP = self.tokenizer.sep_token_id
        self.UNK = self.tokenizer.unk_token_id
        
        # BERT MAX LEN
        self.BML = 512
        # Word MAX LEN
        self.WML = 386
        
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, index):
        words = self.sents[index].split()
        word_length = len(words)
        
        transform_matrix = np.zeros((self.WML,self.BML,), dtype=np.float32)
        
        all_pieces = [self.CLS]
        transform_matrix[0,len(all_pieces)-1] = 1.0
        all_spans = []
        
        for idx, word in enumerate(words):
            tokens = self.tokenizer.tokenize(word)
            pieces = self.tokenizer.convert_tokens_to_ids(tokens)
            if len(pieces) == 0:
                pieces = [self.UNK]
            start = len(all_pieces)
            all_pieces += pieces
            end = len(all_pieces)
            all_spans.append([start, end])

            if len(pieces) != 0:
                piece_num = len(pieces)
                mean_matrix = np.full((piece_num), 1.0/piece_num)
                transform_matrix[idx+1,start:end] = mean_matrix
                
        all_pieces.append(self.SEP)
        cls_text_sep_length = len(all_pieces)
        transform_matrix[len(words),cls_text_sep_length-1] = 1.0
        assert len(all_pieces) <= self.BML
        
        pad_len = self.BML - len(all_pieces)
        all_pieces += [self.PAD] * pad_len
        attention_mask = [1.0] * cls_text_sep_length + [0.0] * pad_len
        assert len(all_pieces) == self.BML
        
        if flag:
            label = [-1]
        elif self.type_label == 'sentiment':
            label = self.label2index[self.sentiments[index]]
        elif self.type_label == 'topic':
            label = self.label2index[self.topics[index]]
        
        return {
            'sent': self.sents[index],
            'sentiment': self.sentiments[index],
            'topic': self.topics[index],
            'attention_mask': attention_mask,
            'transform_matrix': transform_matrix,
            'indices': all_pieces,
            'bert_length': cls_text_sep_length,
            'word_length': word_length,
            'target': label,
        }
    
    @staticmethod
    def pack(items):
        return {
            k: TS_TYPE([x[k] for x in items])
            for k, TS_TYPE in TENSOR_TYPES.items()
        }