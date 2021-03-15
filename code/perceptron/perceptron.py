import numpy as np
from common import *
from data import *

# [0]: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# [1]: https://karpathy.github.io/neuralnets/

def train(X, Y, iterations = 1000, eta = 0.1):
  X = X.T
  Y = Y.T

  # initialize random weight vector with weights in range [-0.5, 0.55]
  w = np.random.random_sample((X.shape[0], 1)) - 0.5

  # training loop
  for i in range(iterations):
    # calculate the current score
    score_i = score(X, w)

    if i % 1000 == 0:
      print(f"Still training... iteration {i}")

    # see "delta rule" from [0]. this is a simplified + numpy'd version of it
    # the constant factors don't matter so I've ommited them
    feature_derivs = np.dot(X, (score_i - Y).T) / Y.size

    # adjust the score based on the simplified gradient vector
    w = w - eta * feature_derivs

  return w


def score(X, w):
  """
  Multiply features by feature weights, sum them up, and run through sigmoid/logistic function to map to [0,1]
  """
  return sigmoid(np.dot(w.T, X))


def predict(scores):
  """
  A score of at least 0.5 indicates a positive review
  A score of less than 0.5 indicates a negative review
  """
  return np.rint(scores)


def test(Y, predicted):
  true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
  for result in np.dstack((Y.T, predicted))[0]:
    [real, predicted] = list(result)

    if real not in [0, 1] or predicted not in [0, 1]:
      print(f"ERROR: INVALID TEST RESULT ARRAY")
    elif real == 1 and predicted == 1:
      true_positive += 1
    elif real == 1 and predicted == 0:
      false_negative += 1
    elif real == 0 and predicted == 1:
      false_positive += 1
    elif real == 0 and predicted == 0:
      true_negative += 1

  accuracy = (true_positive + true_negative) / Y.size
  recall = true_positive / (true_positive + false_negative)
  precision = true_positive / (true_positive + false_positive)
  f1 = 2 * (precision * recall) / (precision + recall)
  print(f"true_positive : {true_positive}")
  print(f"false_positive: {false_positive}")
  print(f"true_negative : {true_negative}")
  print(f"false_negative: {false_negative}")
  print(f"accuracy      : {accuracy}")
  print(f"recall        : {recall}")
  print(f"precision     : {precision}")
  print(f"f1            : {f1}")
