import collections
import json
from KaggleWord2VecUtility import KaggleWord2VecUtility
from nn_lstm_attempt_1 import LSTM
import nn_tools
import torch
import torchtext
from tqdm import tqdm

MODEL_FILE_PATH = "../../nlpModels/64hidden_1layer_0dropout_bidirectional_Word2Vec.pt"
WORD2VEC_VECTORS_FILE_PATH = "../../word2vecModels/300features_40minwords_10context_w2vformat.txt"
WORD2VEC_WORD_FREQ_FILE_PATH = "../../word2vecModels/300features_40minwords_10context_word_freq.json"
BATCH_SIZE = 64
NUM_FEATURES = 300
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0
BIDIRECTIONAL = True


def getAllProbabilityEstimates(lstm, iterator):
    result = torch.Tensor()
    lstm.eval()
    with torch.no_grad():
        for batch in tqdm(iterator):
            text, text_lengths = batch.review

            predictions = lstm(text, text_lengths).squeeze()
            result = torch.cat((result, predictions))

    return result


if __name__ == '__main__':
    text = torchtext.data.Field(tokenize=KaggleWord2VecUtility.review_to_wordlist, batch_first=True, include_lengths=True)
    label = torchtext.data.LabelField(dtype=torch.float, batch_first=True)

    fields = [(None, None), ("sentiment", label), ("review", text)]

    # load datasets
    print("LOADING data from CSV")
    train_data = torchtext.data.TabularDataset(path="../../dataset/processed/A.tsv", format="tsv", fields=fields, skip_header=True)
    dev_data = torchtext.data.TabularDataset(path="../../dataset/processed/B.tsv", format="tsv", fields=fields, skip_header=True)

    print("LOADING Word2Vec Model")
    vectors = torchtext.vocab.Vectors(WORD2VEC_VECTORS_FILE_PATH)
    with open(WORD2VEC_WORD_FREQ_FILE_PATH) as word_freq_file:
        word_counter = collections.Counter(json.load(word_freq_file))
    vocab = torchtext.vocab.Vocab(word_counter, vectors=vectors)
    text.vocab = vocab

    label.build_vocab(train_data)

    # load iterator
    dev_iterator = torchtext.data.BucketIterator(dev_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.review), train=False, sort_within_batch=True)

    true_labels = torch.Tensor()
    for batch in dev_iterator:
        true_labels = torch.cat((true_labels, batch.sentiment))

    # load the model
    lstm = LSTM(NUM_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL, len(text.vocab))
    lstm.load_state_dict(torch.load(MODEL_FILE_PATH))

    # get predictions
    probability_estimates = getAllProbabilityEstimates(lstm, dev_iterator)

    # get ROC data
    FPRs, FNRs, thresholds = nn_tools.TabulateModelPerformanceForROCFromProbabilityEstimates(true_labels, probability_estimates)
    print ("FPRs:", FPRs)
    print ("FNRs:", FNRs)
    print ("Thresholds:", thresholds)
