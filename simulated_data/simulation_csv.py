import csv
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
import numpy as np
# from ast import literal_eval as make_tuple

if __name__ == "__main__":
    number = 2
    with open('_simulation'+str(number), 'w', newline='') as myfile:
        mu = np.array([[1.7], [2.0]])
        alpha = np.array([[0.0, 0.0], [-0.3, -0.2]])
        beta = np.array([[0.0], [1.2]])
        max_jumps = 5000

        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

        for i in range(25):
            np.random.seed(i)

            hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)
            hawkes.simulate()

            tList = hawkes.timestamps
            wr.writerow(tList)

    # with open('_simulation0', 'r') as read_obj:
    #     csv_reader = csv.reader(read_obj)
    #     # Iterate over each row in the csv using reader object
    #     for row in csv_reader:
    #         # row variable is a list that represents a row in csv
    #         print([make_tuple(i) for i in row[0:2]])