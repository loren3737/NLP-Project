from data import *
import perceptron
import os

import pandas as pd

if __name__ == '__main__':
  dataset_A = pd.read_csv( "../../dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
  dataset_B = pd.read_csv( "../../dataset/processed/B.tsv", header=0, delimiter="\t", quoting=3 )
  dataset_C = pd.read_csv( "../../dataset/processed/C.tsv", header=0, delimiter="\t", quoting=3 )
  # dataset_D = pd.read_csv( "../../dataset/processed/D.tsv", header=0, delimiter="\t", quoting=3 )
  # dataset_E = pd.read_csv( "../../dataset/processed/E.tsv", header=0, delimiter="\t", quoting=3 )
  print("Loaded data.")

  training_set = list(dataset_A.to_numpy())
  dev_set = list(dataset_B.to_numpy())

  X_train, Y_train = reviews_to_features(training_set)
  print("Featurized training data.")

  weights = perceptron.train(X_train, Y_train, iterations = 10000)
  print("Done training.")

  X_test, Y_test = reviews_to_features(dev_set)
  print("Featurized test data.")

  test_scores = perceptron.score(X_test.T, weights)
  test_sentiments = perceptron.predict(test_scores)
  perceptron.test(Y_test, test_sentiments)
