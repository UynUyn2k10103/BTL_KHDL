from models.bert_layer import BertEmbedding
import torch
from torch.autograd import Variable


class LSTMModel(torch.nn.Module):
    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.device = args.device
        self.c = len(args.label2index)
        self.embeddings = BertEmbedding(args)

        self.hidden_size = 1024
        self.num_layers = 2
#         self.rnn = nn.RNN(self.embeddings.output_size, self.hidden_dim, self.layer_dim, batch_first=True, nonlinearity='relu')
        self.lstm = torch.nn.LSTM(input_size=self.embeddings.output_size, hidden_size=self.hidden_size,
                                  num_layers=self.num_layers, batch_first=True, dropout=0.25)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, 512),
            torch.nn.Dropout(),
            torch.nn.Tanh(),
            torch.nn.Linear(512, self.c)
        )

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)

        h_1 = Variable(torch.zeros(
            self.num_layers, embeddings.size(0), self.hidden_size).to(self.device))

        c_1 = Variable(torch.zeros(
            self.num_layers, embeddings.size(0), self.hidden_size).to(self.device))
#         print(f'Shape h_1: {h_1.shape}')
#         print(f'Shape c_1: {c_1.shape}')

        _, (hn, cn) = self.lstm(embeddings, (h_1, c_1))
#         print(f'Shape out _: {_.shape}')

        #print("hidden state shpe is:",hn.size())
        y = hn.view(-1, self.hidden_size)

#         print(f'Shape y: {y.shape}')

        final_state = hn.view(
            self.num_layers, embeddings.size(0), self.hidden_size)[-1]
#         print(f'Shape final_state: {final_state.shape}')

        logits = self.fc(final_state)
#         print(f'Shape logits: {logits.shape}')
        preds = torch.argmax(logits, dim=-1)
        return logits, preds
