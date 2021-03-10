import torch

class LSTM(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, dropout,
                 bidirectional, num_embeddings):
        super().__init__()

        # embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings, num_features)

        # LSTM layer
        self.lstm = torch.nn.LSTM(num_features, hidden_size,
                                  num_layers=num_layers, batch_first=True,
                                  dropout=dropout, bidirectional=bidirectional)

        # output layer with single output
        # TODO input num features may need to be changed if LSTM is
        # bidirectional
        self.outputLayer = torch.nn.Sequential(torch.nn.Linear(hidden_size, 1),
                                               torch.nn.Sigmoid())

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)

        # pack padding so that all sequences (reviews) are the same length
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        
        _, (hidden_state, _) = self.lstm(packed)

        # use the hidden state of the last layer
        out = hidden_state[-1, :, :]
        
        out = self.outputLayer(out)

        return out
