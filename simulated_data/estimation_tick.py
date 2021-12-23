import numpy as np
import csv
from ast import literal_eval as make_tuple
from class_and_func.streamline_tick import four_estimation
from dictionary_parameters import dictionary as param_dict


if __name__ == "__main__":
    dim = 2
    number = 2
    theta = param_dict[number]
    beta = theta[-dim:].reshape((dim,1)) + 1e-16
    print(beta)

    C = 1
    with open('_simulation'+str(number), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        with open('_estimation'+str(number)+"tick", 'w', newline='') as myfile:
            i = 1
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for row in csv_reader:
                print("# ", i)
                tList = [make_tuple(i) for i in row]

                params_tick = four_estimation(beta, tList, penalty="l2")
                print(params_tick)
                i += 1
                break

