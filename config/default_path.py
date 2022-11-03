import torch

name_model = 'gru_best.pth'
path_label = 'dataset/_UIT-VSFC/label2index.json'
type_label = 'topic'
version = 'gru'
bert_type = 'vinai/phobert-base'
flag = True
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'