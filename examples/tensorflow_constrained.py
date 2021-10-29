import numpy as np
from matplotlib import pyplot as plt
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
import tensorflow as tf
import tensorflow_constrained_optimization as tfco
from class_and_func.likelihood_functions import multivariate_loglikelihood_tf, multivariate_loglikelihood_simplified


if __name__ == "__main__":
    seed = 7
    tf.random.set_seed(seed)
    np.random.seed(seed)

    dim = 2  # 2, 3 ou 4
    lr = 0.01
    nb_epoch = 100

    mu = np.array([[0.5], [1.0]])
    alpha = np.array([[-0.9, 3], [1.2, 1.5]])
    beta = np.array([[4], [5]])

    max_jumps = 500

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    hawkes.simulate()

    tList = hawkes.timestamps

    mutf = tf.Variable(tf.random.uniform(shape=[dim,1]))
    alphatf = tf.Variable(tf.random.uniform(shape=[dim, dim]))
    betatf = tf.Variable(tf.random.uniform(shape=[dim, 1]))

    print(mu)
    print("np",
          multivariate_loglikelihood_simplified((mu, alpha, beta), tList))
    print("torch",
          multivariate_loglikelihood_tf((mutf, alphatf, betatf), tList))

