import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


if __name__ == "__main__":

    np.random.seed()

    poiss = np.random.poisson(100000)

    theta_vect = 2*np.pi*np.random.uniform(0, 1, poiss)
    r_vect = np.sqrt(np.random.uniform(0, 1, poiss))

    x_vect = r_vect*np.cos(theta_vect)
    y_vect = r_vect*np.sin(theta_vect)

    choice = np.random.choice(poiss, size=10000, replace=False)

    x_after = x_vect[choice]
    y_after = y_vect[choice]

    fig, ax = plt.subplots()
    ax.scatter(x_vect, y_vect)
    ax.scatter(x_after, y_after)

    choice = np.random.choice(poiss, size=100, replace=False)

    x_after = x_vect[choice]
    y_after = y_vect[choice]

    ax.scatter(x_after, y_after)

    # fig, ax = plt.subplots()
    # stats.probplot(x_after, dist=stats.uniform, plot=ax)

    plt.show()

