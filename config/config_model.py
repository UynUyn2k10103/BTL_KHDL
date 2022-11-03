from transformers import AutoModel, AutoTokenizer
from config import default_path



# load phobert pretrain model
tokenizer = AutoTokenizer.from_pretrained(default_path.bert_type)
bert_model = AutoModel.from_pretrained(default_path.bert_type)
