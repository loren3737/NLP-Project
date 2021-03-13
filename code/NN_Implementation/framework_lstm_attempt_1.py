import collections
from KaggleWord2VecUtility import KaggleWord2VecUtility
from nn_lstm_attempt_1 import LSTM
import nn_tools
import pyhelpers
import torch
import torchtext
from tqdm import tqdm

### CONFIGURATION ###################
BATCH_SIZE = 64
NUM_FEATURES = 300
HIDDEN_SIZE = 32
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = False

USE_WORD2VEC = True
WORD2VEC_VECTORS_FILE_PATH = "../../word2vecModels/300features_40minwords_10context_w2vformat.txt"
WORD2VEC_WORD_FREQ_FILE_PATH = "../../word2vecModels/300features_40minwords_10context_word_freq.json"
#####################################


def trainOneEpoch(lstm, iterator, optimizer, loss_function):
    total_loss = 0
    lstm.train()
    for batch in tqdm(iterator):
        text, text_lengths = batch.review

        predictions = lstm(text, text_lengths).squeeze()

        loss = loss_function(predictions, batch.sentiment)

        # Reset the gradients in the network to zero
        optimizer.zero_grad()

        # Backprop the errors from the loss on this iteration
        loss.backward()

        # Do a weight update step
        optimizer.step()

        total_loss += loss.item()

    print ("Average loss:", total_loss / len(iterator))


def evaluate(lstm, iterator, classification_threshold=0.5):
    total_accuracy = 0
    lstm.eval()
    with torch.no_grad():
        for batch in tqdm(iterator):
            text, text_lengths = batch.review

            predictions = lstm(text, text_lengths).squeeze()

            # TODO use classification threshold
            rounded_predictions = torch.round(predictions)

            accuracy = nn_tools.AccuracyFromTensors(batch.sentiment, rounded_predictions)

            total_accuracy += accuracy.item()

    return total_accuracy / len(iterator)


if __name__ == '__main__':
    # preprocess data
    text = torchtext.data.Field(tokenize=KaggleWord2VecUtility.review_to_wordlist, batch_first=True, include_lengths=True)
    label = torchtext.data.LabelField(dtype=torch.float, batch_first=True)

    labeled_fields = [(None, None), ("sentiment", label), ("review", text)]
    unlabeled_fields = [(None, None), ("review", text)]

    # load datasets
    print("LOADING data from CSV")
    train_data = torchtext.data.TabularDataset(path="../../dataset/processed/A.tsv", format="tsv", fields=labeled_fields, skip_header=True)
    dev_data = torchtext.data.TabularDataset(path="../../dataset/processed/B.tsv", format="tsv", fields=labeled_fields, skip_header=True)
    dataset_D = torchtext.data.TabularDataset(path="../../dataset/processed/D.tsv", format="tsv", fields=unlabeled_fields, skip_header=True)
    dataset_E = torchtext.data.TabularDataset(path="../../dataset/processed/E.tsv", format="tsv", fields=unlabeled_fields, skip_header=True)
    print ("Training example:", vars(train_data.examples[0]))


    # prep input and output sequences
    if USE_WORD2VEC:
        print("LOADING Word2Vec Model")
        
        vectors = torchtext.vocab.Vectors(WORD2VEC_VECTORS_FILE_PATH)

        word_counter = collections.Counter(pyhelpers.store.load_json(WORD2VEC_WORD_FREQ_FILE_PATH, verbose=True))

        vocab = torchtext.vocab.Vocab(word_counter, vectors=vectors)

        text.vocab = vocab
    else:
        # use GloVe pretrained embeddings
        print ("LOADING GloVe pretrained embeddings")
        text.build_vocab(train_data, dataset_D, dataset_E, min_freq=40, vectors = "glove.6B.300d")
        
    label.build_vocab(train_data)

    print ("Size of text vocab:", len(text.vocab))
    print ("Size of label vocab:", len(label.vocab))
    print ("Commonly used words:", text.vocab.freqs.most_common(10))
    # print ("Word dictionary:", text.vocab.stoi)

    # load iterators
    train_iterator, dev_iterator = torchtext.data.BucketIterator.splits((train_data, dev_data), batch_size=BATCH_SIZE, sort_key=lambda x: len(x.review), sort_within_batch=True)


    print("TRAINING LSTM model")
    
    # instantiate model
    lstm = LSTM(NUM_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL, len(text.vocab))

    # initialize pretrained embeddings from word2vec or GloVe
    lstm.embedding.weight.data.copy_(text.vocab.vectors)

    # define optimizer and loss function
    optimizer = torch.optim.Adam(lstm.parameters())
    loss_function = torch.nn.BCELoss()

    epoch = 0
    print ("Epoch:", epoch)
    print ("Dev set accuracy:", evaluate(lstm, dev_iterator))
    # TODO automate convergence
    user_input = str(input("Continue training? y/n: "))
    while user_input.lower() == "y":
        trainOneEpoch(lstm, train_iterator, optimizer, loss_function)
        epoch += 1

        print ("Epoch:", epoch)
        print ("Dev set accuracy:", evaluate(lstm, dev_iterator))
        user_input = str(input("Continue training? y/n"))
