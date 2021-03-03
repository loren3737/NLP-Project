import time
import numpy
import torch 
import nn_model
import nn_tools
import KaggleWord2VecUtility
from gensim.models import Word2Vec
from gen_Doc2Vec import generate_doc2vec
from gen_Word2Vev import generate_word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import pandas as pd
from KaggleAverageVector import getAvgFeatureVecs, getCleanReviews

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

### CONFIGURATION ###################
generate_embedding = False
word2vec_name = "300features_40minwords_10context"
doc2vec_name = "doc2vec_300features_40minwords_10context"
NUM_FEATURES = 300
#####################################

print()
print("Sentiment Analysis - Neural Network using Word2Vec")
print()

# Step 1 Load in data
print("LOADING data from CSV")
dataset_A = pd.read_csv( "NLP-Project/dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
dataset_D = pd.read_csv( "NLP-Project/dataset/processed/D.tsv", header=0, delimiter="\t", quoting=3 )
dataset_E = pd.read_csv( "NLP-Project/dataset/processed/E.tsv", header=0, delimiter="\t", quoting=3 )
# dataset_A = pd.read_csv( "../../dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
# dataset_D = pd.read_csv( "../../dataset/processed/D.tsv", header=0, delimiter="\t", quoting=3 )
# dataset_E = pd.read_csv( "../../dataset/processed/E.tsv", header=0, delimiter="\t", quoting=3 )
datasets = [dataset_A, dataset_D, dataset_E]

# Step 2 Generate or Load Word2Vec/Doc2Vec

if generate_embedding:
    print("CREATING Word2Vec Model")
    word2vec_model = generate_word2vec(word2vec_name, datasets)
else:
    try:
        print("LOADING Word2Vec Model")
        word2vec_model = Word2Vec.load(word2vec_name)
    except Exception as ex:
        print(ex)
        print("Couldn't load Word2Vec, BUILDING...")
        word2vec_model = generate_word2vec(word2vec_name, datasets)

if generate_embedding:
    print("CREATING Doc2Vec Model")
    doc2vec_model = generate_doc2vec(doc2vec_name, datasets)
else:
    try:
        print("LOADING Doc2Vec Model")
        doc2vec_model = Doc2Vec.load(doc2vec_name)
    except Exception as ex:
        print(ex)
        print("Could't load Doc2Vec, BUILDING...")
        doc2vec_model = generate_doc2vec(doc2vec_name, datasets)
        
# Step 3 Gather Word2Vec Features
print("FEATURIZING")
xTrain = getAvgFeatureVecs(getCleanReviews(dataset_A), word2vec_model, NUM_FEATURES)
xTrain = torch.tensor(xTrain)
yTrain = torch.Tensor([y_val for y_val in dataset_A["sentiment"]])

# Step 5 Training NN model
print("TRAINING nn model")
l_model = nn_model.NeuralNetwork()
l_model.train_model_persample(xTrain, yTrain)
yValidatePredicted = l_model.predict(xTrain)

print(yValidatePredicted)