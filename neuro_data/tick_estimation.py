import numpy as np
import csv
from class_and_func.streamline_tick import four_estimation_with_grid
import pickle
import time


def obtain_average_estimation(file_name, number, dim, number_estimations):
    n = 0
    if file_name[0:4] == "tick":
        if file_name[5:9] == "beta":
            result = np.zeros((dim + dim * dim * dim,))
        else:
            result = np.zeros((dim + dim * dim,))
    else:
        result = np.zeros((2 * dim + dim * dim,))
    with open("estimation/_traitements2_" + str(number) + file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if n < number_estimations:
                result += np.array([float(i) for i in row])
                n += 1
    result /= n

    return result


if __name__ == "__main__":
    C_def = 0.01

    np.random.seed(0)

    beta_grid = np.array([0.01, 0.1, 1, 5, 10, 50])

    for number in range(1, 2):
        a_file = open("traitements2/train" + str(number) + ".pkl", "rb")
        tList, filtre_dict_orig, orig_dict_filtre = pickle.load(a_file)
        print(tList[0], tList[-1])
        dim = len(filtre_dict_orig)
        a_file.close()

        estimation = obtain_average_estimation("grad", number, dim, 1)
        beta = estimation[-dim:].reshape((dim, 1))

        file_simple = open("estimation/_traitements2_"+str(number)+"tick", 'w', newline='')
        file_pen = open("estimation/_traitements2_"+str(number)+"tick_bfgs", 'w', newline='')
        file_beta_simple = open("estimation/_traitements2_"+str(number)+"tick_beta", 'w', newline='')
        file_beta_pen = open("estimation/_traitements2_"+str(number)+"tick_beta_bfgs", 'w', newline='')

        ws = csv.writer(file_simple, quoting=csv.QUOTE_ALL)
        wp = csv.writer(file_pen, quoting=csv.QUOTE_ALL)
        wbs = csv.writer(file_beta_simple, quoting=csv.QUOTE_ALL)
        wbp = csv.writer(file_beta_pen, quoting=csv.QUOTE_ALL)

        start_time = time.time()
        params_tick = four_estimation_with_grid(beta, beta_grid, tList, penalty="l2", C=C_def)
        end_time = time.time()
        print("Time it took: ", (end_time - start_time)//60, " minutes.")

        ws.writerow(np.concatenate((params_tick[0][0], params_tick[0][1].ravel())).tolist())
        wp.writerow(np.concatenate((params_tick[1][0], params_tick[1][1].ravel())).tolist())
        wbs.writerow(np.concatenate((params_tick[2][0], params_tick[2][1].ravel())).tolist())
        wbp.writerow(np.concatenate((params_tick[3][0], params_tick[3][1].ravel())).tolist())

        file_simple.close()
        file_pen.close()
        file_beta_simple.close()
        file_beta_pen.close()