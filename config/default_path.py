import torch

name_model = 'mlp_best.pth'
path_label = 'dataset/_UIT-VSFC/label2index.json'
type_label = 'topic'
version = 'mlp'
bert_type = 'vinai/phobert-base'
flag = True
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
