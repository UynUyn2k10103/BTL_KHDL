from models.bert_layer import BertEmbedding
import torch

class GRUModel(torch.nn.Module):
    def __init__(self, args):
        super(GRUModel, self).__init__()
        self.device = args.device
        self.c = len(args.label2index)
        self.embeddings = BertEmbedding(args)
        
        
        self.hidden_size = 1024
        self.num_layers = 2
        
        self.gru = torch.nn.GRU(input_size=self.embeddings.output_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True,dropout = 0.25)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, 512),
            torch.nn.Dropout(),
            torch.nn.Tanh(),
            torch.nn.Linear(512, self.c)
        )

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)
       
        out, hidden = self.gru(embeddings)
        
        out = out[:, -1, :]
        
        logits = self.fc(out)
        preds = torch.argmax(logits, dim=-1)
        
        return logits, preds
        