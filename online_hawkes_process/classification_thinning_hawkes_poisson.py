from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import kstest


def classification_noised_hawkes(mu, alpha, beta, noise, timestamps):
    # Timestamps without first nor last (as they are 0 marked)
    # Only excitation
    # We use the opposite of the loglikelihood so Lambda - sum(lambda)
    classes = np.zeros(len(timestamps))
    l_hawkes = 0
    l_poisson = 0
    ic_hawkes = mu
    tb = 0

    for i, tc in enumerate(timestamps):
        l_hawkes += mu * (tc - tb) + (1 / beta) * (ic_hawkes - mu) * (1 - np.exp(-beta * (tc - tb)))
        ic_hawkes = mu + (ic_hawkes - mu) * np.exp(-beta * (tc -tb))
        l_poisson += noise * (tc - tb)
        u = np.random.rand()
        if u < l_poisson / (l_poisson + l_hawkes):
            classes[i] = 2
            l_poisson -= np.log(noise)
        else:
            classes[i] = 1
            l_hawkes -= np.log(ic_hawkes)
            ic_hawkes += alpha
        tb = tc
    return(classes)


if __name__ == "__main__":
    sns.set_theme()

    np.random.seed(0)

    noise = 0.1
    mu = np.array([[1.0], [noise]])
    alpha = np.array([[0.5, 0.0],
                      [0.0, 0.0]])
    beta = np.array([[1.0], [1.0]])

    hp = multivariate_exponential_hawkes(mu, alpha, beta, max_time=50)
    hp.simulate()
    #hp.plot_intensity()

    times = np.array([t for t, _ in hp.timestamps[1:-1]])
    real_marks = np.array([m for _, m in hp.timestamps[1:-1]])

    nb_times = len(real_marks)
    aux = np.zeros(nb_times)

    repet = 10000

    for _ in range(repet):
        aux += classification_noised_hawkes(mu[0], alpha[0, 0], beta[0], noise, times)

    aux /= repet

    estimated_marks = np.array([2 if x >= 1.5 else 1 for x in aux])

    fig, ax = plt.subplots()
    ax.scatter(range(nb_times), real_marks, s=30)
    ax.scatter(range(nb_times), estimated_marks, s=10, alpha=0.5)

    print("Hawkes error: ", np.mean(np.abs(real_marks[real_marks == 1] - estimated_marks[real_marks == 1])))
    print("Poisson error: ", np.mean(np.abs(real_marks[real_marks == 2] - estimated_marks[real_marks == 2])))
    print("Total error: ", np.mean(np.abs(real_marks - estimated_marks)))

    plt.show()

