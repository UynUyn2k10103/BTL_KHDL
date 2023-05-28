import tqdm
from config.config_model import tokenizer
from config.config_args import args
from torch.utils.data import DataLoader
from models.load_model import load_model
from dataset.load_label import load_label
from dataset.load_dataset import VSFC_DataSet
from db.save_db import insert_data
from datetime import date


model = load_model()
labels = load_label()


def predict(sentence):

    input_dataset = VSFC_DataSet(sentence, tokenizer, args, False)
    input_dl = DataLoader(input_dataset,
                          batch_size=args.batch_size,
                          num_workers=2,
                          shuffle=False,
                          collate_fn=VSFC_DataSet.pack)

    model.eval()

    for batch in tqdm.tqdm(input_dl, desc='Predict....'):

        logits, preds = model(batch)
        preds = preds.cpu().numpy().tolist()

    insert_data(sentence=sentence, label=preds[0], joining=date.today())
    return {'answer': labels[preds[0]]}
    # return labels
