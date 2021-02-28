from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from KaggleWord2VecUtility import KaggleWord2VecUtility
import logging
import nltk.data
import pandas as pd

# Set values for various parameters. We may wanna tune these
NUM_FEATURES = 300    # Word vector dimensionality
MIN_WORD_COUNT = 40   # Minimum word count
NUM_WORKERS = 4       # Number of threads to run in parallel
CONTEXT = 10          # Context window size

MODEL_NAME = "doc2vec_300features_40minwords_10context"

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



    # ****** Split the training sets into clean docs
    #
    # flatten each list of sentences from a review into a single list of words
    docs = []  # Initialize an empty list of docs

    print ("Parsing sentences from dataset A")
    for review in dataset_A["review"]:
        sentences = KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        docs.append([word for sentence in sentences for word in sentence])

    print ("Parsing sentences from dataset D")
    for review in dataset_D["review"]:
        sentences = KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        docs.append([word for sentence in sentences for word in sentence])

    print ("Parsing sentences from dataset E")
    for review in dataset_E["review"]:
        sentences = KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        docs.append([word for sentence in sentences for word in sentence])

    # ****** Set parameters and train the doc2vec model
    #
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Initialize and train the model
    print ("Training Doc2Vec model...")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
    model = Doc2Vec(documents, workers=NUM_WORKERS, \
                vector_size=NUM_FEATURES, min_count = MIN_WORD_COUNT, \
                window = CONTEXT, seed=1)

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    # You can load the model later using Doc2Vec.load()
    model.save(MODEL_NAME)
