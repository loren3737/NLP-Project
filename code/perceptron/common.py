import numpy as np
from multiprocessing import cpu_count

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def get_num_processes(data_size):
  optimal = 8

  if data_size <= 5000:
    optimal = 4

  return min(optimal, cpu_count() - 1)
