# Before running this, remember to download the punkt data:
# nltk.download('punkt')

from gensim.models import Word2Vec
from KaggleWord2VecUtility import KaggleWord2VecUtility
import logging
import nltk.data
import pandas as pd

# Set values for various parameters. We may wanna tune these
NUM_FEATURES = 300    # Word vector dimensionality
MIN_WORD_COUNT = 40   # Minimum word count
NUM_WORKERS = 4       # Number of threads to run in parallel
CONTEXT = 10          # Context window size
DOWNSAMPLING = 1e-3   # Downsample setting for frequent words

MODEL_NAME = "300features_40minwords_10context"

if __name__ == '__main__':

    # Read data from files
    dataset_A = pd.read_csv( "../../dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
    dataset_D = pd.read_csv( "../../dataset/processed/D.tsv", header=0, delimiter="\t", quoting=3 )
    dataset_E = pd.read_csv( "../../dataset/processed/E.tsv", header=0, delimiter="\t", quoting=3 )

    # Verify the number of reviews that were read (95,000 in total)
    print ("Read %d labeled A reviews, %d unlabeled D reviews, " \
     "and %d unlabeled E reviews\n" % (dataset_A["review"].size,
     dataset_D["review"].size, dataset_E["review"].size ))



    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



    # ****** Split the training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences

    print ("Parsing sentences from dataset A")
    for review in dataset_A["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    print ("Parsing sentences from dataset D")
    for review in dataset_D["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    print ("Parsing sentences from dataset E")
    for review in dataset_E["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Initialize and train the model (this will take some time)
    print ("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=NUM_WORKERS, \
                size=NUM_FEATURES, min_count = MIN_WORD_COUNT, \
                window = CONTEXT, sample = DOWNSAMPLING, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # You can load the model later using Word2Vec.load()
    model.save(MODEL_NAME)
