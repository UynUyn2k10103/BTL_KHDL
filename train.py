from my_utils.metrics import metrics
from my_utils.evaluate import evaluate
from models.lstm_models import LSTMModel
from dataset.load_dataset import *
from config.config_model import bert_model, tokenizer
from config.config_args import args
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import tqdm
import os



#save the best version of model
def save_model(model, path, name):
    os.makedirs(os.path.join(path), exist_ok = True)
    torch.save(model.state_dict(), os.path.join(path, name))

## prepare data to train
train_dataset = VSFC_DataSet('train',tokenizer, args)
dev_dataset = VSFC_DataSet('dev',tokenizer, args)
test_dataset = VSFC_DataSet('test',tokenizer, args)

train_dl = DataLoader(train_dataset,
                      batch_size=args.batch_size,
                      num_workers=2,
                      shuffle=True,
                      collate_fn=VSFC_DataSet.pack)

dev_dl = DataLoader(dev_dataset,
                    batch_size=args.batch_size,
                    num_workers=2,
                    shuffle=False,
                    collate_fn=VSFC_DataSet.pack)

test_dl = DataLoader(test_dataset,
                     batch_size=args.batch_size,
                     num_workers=2,
                     shuffle=False,
                     collate_fn=VSFC_DataSet.pack)

#Config model
model = LSTMModel(args).to(args.device)

params = [x for x in model.parameters() if x.requires_grad]
optimizer = torch.optim.Adam(params, lr=args.lr)
ce = CrossEntropyLoss()

#train model
global_iter = 0
best_dev = {'r': 0, 'p': 0, 'f': 0} 
for epoch in range(args.epoch):
        model.train()
        bar = tqdm.tqdm(train_dl, desc='Training', total=len(train_dl))
        for batch in bar:
            global_iter += 1
            logits, preds = model(batch)
#             print(batch['target'].shape)
#             print(f'Kích thước logits: {logits.shape}, kích thước preds: {preds.shape}')

            loss = ce(logits, batch['target'].to(args.device))
#             print(logits.shape)

            if global_iter % 10 == 0:
                l = loss.detach().cpu().numpy()
                # experiment.log_metric("Train_loss", l, step = global_iter, epoch = epoch)
                bar.set_description(f'Training: Loss={l:.4f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        dev_perf = evaluate(model, dev_dl, args, 'Dev', global_iter)
        # experiment.log_metric("F1_score", dev_perf['f'], step = global_iter, epoch = epoch)
        
        if dev_perf['f'] > best_dev['f']:
            best_dev = dev_perf
            print('New best @ {}'.format(epoch))
            save_model(model, 'checkpoints', f'{args.version}_best.pth')
