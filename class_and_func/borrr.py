import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    lambda_0 = 1
    alpha = 2
    beta = 1

    lambda_1 = lambda_0 +


    t = 0
    flag = t < max_time

    while flag:
        upper_intensity = max(lambda_0,
                              lambda_0 + aux * np.exp(-beta * (t - timestamps[-1])))

        t += np.random.exponential(1 / upper_intensity)
        candidate_intensity = lambda_0 + aux * np.exp(-beta * (t - timestamps[-1]))

        flag = t < max_time

        if upper_intensity * np.random.uniform() <= candidate_intensity and flag:
            timestamps += [t]
            intensity_jumps += [candidate_intensity + alpha]
            aux = aux * np.exp(-beta * (t - timestamps[-2])) + alpha
