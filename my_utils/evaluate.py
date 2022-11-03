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
