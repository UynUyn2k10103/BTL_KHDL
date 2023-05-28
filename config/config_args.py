from config import config_labels
from config import default_path


class Args():
    lr = 2e-5
    epoch = 16
    batch_size = 16
    update_bert = False
    bert_type = default_path.bert_type
    device = default_path.device
    type_label = default_path.type_label
    version = default_path.version
    label2index = config_labels.label2index


args = Args()
