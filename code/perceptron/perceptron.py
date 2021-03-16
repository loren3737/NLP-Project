import numpy as np
from common import *
from data import *

# [0]: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# [1]: https://karpathy.github.io/neuralnets/
# [2]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html

def train(X, Y, iterations = 1000, eta = 0.1):
  X = X.T
  Y = Y.T

  # initialize random weight vector with weights in range [-0.5, 0.55]
  w = np.random.random_sample((X.shape[0], 1)) - 0.5

  track_loss_at = np.unique(np.round(np.logspace(np.log10(0.01), 1, num=100) / 10.0 * iterations))
  loss_iterator = iter(track_loss_at)
  next_loss_at = next(loss_iterator)
  losses = []

  # training loop
  for i in range(iterations):
    # calculate the current score
    score_i = score(X, w)


    if i == next_loss_at:
      next_loss_at = next(loss_iterator, -1)

      # log loss equation from [2]
      avg_log_loss = np.sum((Y * np.log(score_i)) + (1.0 - Y) * np.log(1.0 - score_i)) * (-1.0 / Y.size)
      avg_lse_loss = np.sum(0.5 * np.square(Y - score_i)) / Y.size
      losses.append((i, avg_lse_loss))
      print(f"Iteration: {i:04} | Loss: {avg_lse_loss}")

    # see "delta rule" from [0]. this is a simplified + numpy'd version of it
    # the constant factors don't matter so I've ommited them
    # also, the update equation is the same for both log loss and lse loss
    feature_derivs = np.dot(X, (score_i - Y).T) / Y.size

    # adjust the score based on the simplified gradient vector
    w = w - eta * feature_derivs

  return w, losses


def score(X, w):
  """
  Multiply features by feature weights, sum them up, and run through sigmoid/logistic function to map to [0,1]
  """
  return sigmoid(np.dot(w.T, X))[0]


def predict(scores, threshold = 0.5):
  """
  A score of at least threshold indicates a positive review
  A score of less than threshold indicates a negative review
  """
  return np.where(scores > threshold, 1, 0)


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

  total_positives = np.count_nonzero(Y)
  total_negatives = Y.size - total_positives

  accuracy = (true_positive + true_negative) / Y.size
  recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
  precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
  f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

  false_negative_rate = false_negative / total_positives
  false_positive_rate = false_positive / total_negatives


  return (accuracy, recall, precision, f1, false_positive_rate, false_negative_rate)
