from data import *
import perceptron
import os

import pandas as pd

ITERATIONS = int(os.getenv('ITERATIONS', 10000))
ETA = float(os.getenv('ETA', '0.1'))
ENABLE_ROC = bool(int(os.getenv('ROC', 0)))

if __name__ == '__main__':
  dataset_A = pd.read_csv( "../../dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
  dataset_B = pd.read_csv( "../../dataset/processed/B.tsv", header=0, delimiter="\t", quoting=3 )
  dataset_C = pd.read_csv( "../../dataset/processed/C.tsv", header=0, delimiter="\t", quoting=3 )
  # dataset_D = pd.read_csv( "../../dataset/processed/D.tsv", header=0, delimiter="\t", quoting=3 )
  # dataset_E = pd.read_csv( "../../dataset/processed/E.tsv", header=0, delimiter="\t", quoting=3 )

  training_set = list(dataset_A.to_numpy())
  dev_set = list(dataset_B.to_numpy())
  test_set = list(dataset_C.to_numpy())

  print("Loaded data.")

  X_train, Y_train = reviews_to_features(training_set)
  print("Featurized training data.")

  weights, losses = perceptron.train(X_train, Y_train, iterations = ITERATIONS, eta = ETA)
  print("Done training.")

  X_test, Y_test = reviews_to_features(dev_set)
  print("Featurized test data.")

  test_scores = perceptron.score(X_test.T, weights)
  test_sentiments = perceptron.predict(test_scores)
  (accuracy, recall, precision, f1, false_positive_rate, false_negative_rate) = perceptron.test(Y_test, test_sentiments)
  print(f"Predicted scores w/ threshold = 0.5, iterations = {ITERATIONS}, eta = {ETA}:")
  print(f"  - accuracy      : {accuracy}")
  print(f"  - recall        : {recall}")
  print(f"  - precision     : {precision}")
  print(f"  - f1            : {f1}")
  print(f"  - fpr           : {false_positive_rate}")
  print(f"  - fnr           : {false_negative_rate}")

  if ENABLE_ROC:
    fpr = []
    fnr = []
    thresholds = list(np.round(np.linspace(0,1,101), decimals = 2))
    for threshold in thresholds:
      test_sentiments = perceptron.predict(test_scores, threshold = threshold)
      (accuracy, recall, precision, f1, false_positive_rate, false_negative_rate) = perceptron.test(Y_test, test_sentiments)
      fpr.append(false_positive_rate)
      fnr.append(false_negative_rate)

    print(f"fpr = {fpr}")
    print(f"fnr = {fnr}")
    print(f"thresholds = {thresholds}")
    print(f"losses = {losses}")
