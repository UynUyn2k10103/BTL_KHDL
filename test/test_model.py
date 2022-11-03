from models.load_model import load_model
# from my_utils.evaluate import evaluate
from torch.utils.data import DataLoader
from config.config_args import args
from dataset.load_dataset import VSFC_DataSet
from config.config_model import tokenizer

from tqdm import tqdm
from my_utils import metrics

def evaluate(model, dl, args, msg = 'Test', global_iter = 0):
    model.eval()
    all_golds = []
    all_preds = []

    bar = tqdm(dl, desc=msg, total=len(dl))
    for batch in bar:
        golds = batch['target'].numpy().tolist()
        all_golds += golds

        logits, preds = model(batch)
        if args.device == "cuda":
            preds = preds.cpu().numpy().tolist()
        else:
            preds = preds.numpy().to_list()
        all_preds += preds
    
    perfs = metrics.metrics(all_golds, all_preds, labels = None)

    print('{}: {:.2f} {:.2f} {:.2f} '.format(msg,
                                         perfs['p'],
                                         perfs['r'],
                                         perfs['f'],
                                         ))
    
    return perfs


def test():
    test_dataset = VSFC_DataSet('test', tokenizer, args)
    print(test_dataset)
    test_dl = DataLoader(test_dataset,
                        batch_size=args.batch_size,
                        num_workers=2,
                        shuffle=False,
                        collate_fn=VSFC_DataSet.pack)


    model = load_model()

    print('----Start evaluate-----')
    evaluate(model, test_dl, args, 'Test', 0)

if __name__ == '__main__':
    test()
