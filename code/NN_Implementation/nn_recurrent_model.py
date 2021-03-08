import torch.nn as nn
import time
import math

def timeSince(since):
            now = time.time()
            s = now - since
            m = math.floor(s / 60)
            s -= m * 60
            return '%dm %ds' % (m, s)

class RNN(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, layer1, layer2, output_size):
        super(RNN, self).__init__()

        self.initialHidden = hidden1

        self.RecurrentOne = nn.RNN(input_size, hidden1, layer1)
        self.RecurrentTwo = nn.RNN(layer1, hidden2, layer2)
        self.RecurrentOutput = nn.RNN(layer2, 1, 1)

        # Not needed for binary output
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hide=None):

        if hide is None:
            hide = torch.zeros(1, self.initHidden)
        
        out, hide = self.RecurrentOne(input_tensor, hide)
        out, hide = self.RecurrentTwo(out, hide)
        out, hide = self.RecurrentOutput(out, hide)
        return out, hide

    def train_sentence(self, sentence_features, sentiment, learning_rate):

        self.zero_grad()
        hidden = None
        lossFunction = torch.nn.MSELoss(reduction='mean')

        for i in range(sentence_features.size()[0]):
            output, hidden = self(sentence_features[i], hidden)

        loss = lossFunction(output, sentiment)
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in self.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

        return output, loss.item()

    def train(self, xTrain, yTrain, learning_rate=0.1, epochs = 5):
        
        # Metrics
        print_every = 5000
        plot_every = 1000

        # Keep track of losses for plotting
        current_loss = 0
        all_losses = []
        start = time.time()

        for iter in range(1, epochs + 1):
            for index, sentence_features in xTrain:
                sentiment = yTrain[index]

                output, loss = train_sentence(sentence_features, sentiment)
                current_loss += loss