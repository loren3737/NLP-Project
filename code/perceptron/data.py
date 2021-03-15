import numpy as np
from common import *
from multiprocessing import Pool
from gensim.models import Doc2Vec

doc2vec_model_filepath = "../../word2vecModels/doc2vec_300features_40minwords_10context"
doc2vec_model = Doc2Vec.load(doc2vec_model_filepath, mmap='r')
doc2vec_model.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=True)


def infer_vector_worker(review):
  i, (review_id, sentiment, document) = review
  features = doc2vec_model.infer_vector(document.split())
  return i, sentiment, features


def reviews_to_features(reviews):
  num_reviews = len(reviews)
  # X - inputs (reviews -> review d2v features)
  X = np.zeros((num_reviews, doc2vec_model.vector_size))
  # Y - outputs (sentiments)
  Y = np.zeros((num_reviews, 1))

  with Pool(processes=get_num_processes(num_reviews)) as pool:
    features = pool.map(infer_vector_worker, enumerate(reviews))

  for i, sentiment, features in features:
    Y[i, 0] = sentiment
    X[i] = features

  return (X, Y)
