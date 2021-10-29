import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.likelihood_functions import multivariate_loglikelihood_torch, multivariate_loglikelihood_simplified, multivariate_lstsquares_torch
import os


if __name__ == "__main__":
    ### Simulation of event times
    path = "/Users/alisdair/PycharmProjects/hawkes-inhibition/examples/runs/"
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)

    dim = 2  # 2, 3 ou 4
    lr = 0.001
    nb_epoch = 25

    mu = np.array([[0.5], [1.0]])
    alpha = np.array([[-0.9, 3], [1.2, 1.5]])
    beta = np.array([[4], [5]])

    max_jumps = 1000

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    hawkes.simulate()

    tList = hawkes.timestamps

    torch.manual_seed(seed)
    mulog = torch.nn.Parameter(torch.rand(dim, 1, dtype=torch.float64))
    alphalog = torch.nn.Parameter(torch.rand(dim, dim, dtype=torch.float64))
    betalog = torch.nn.Parameter(torch.rand(dim, 1, dtype=torch.float64))

    torch.manual_seed(seed)
    mulst = torch.nn.Parameter(torch.rand(dim, 1, dtype=torch.float64))
    alphalst = torch.nn.Parameter(torch.rand(dim, dim, dtype=torch.float64))
    betalst = torch.nn.Parameter(torch.rand(dim, 1, dtype=torch.float64))

    # print(mu)
    # print("torch", multivariate_loglikelihood_torch((torch.tensor(mu)+45, torch.tensor(alpha), torch.tensor(beta)), tList))
    # print("np",
    #       multivariate_loglikelihood_simplified((mu+45,alpha,beta),tList))

    parameterslog = [mulog, alphalog, betalog]
    parameterslst = [mulst, alphalst, betalst]

    optimlog = torch.optim.SGD(params=parameterslog, lr=lr)
    optimlst = torch.optim.SGD(params=parameterslst, lr=lr)

    os.system('rm -rf ' + path)
    writer = SummaryWriter()
    with torch.autograd.set_detect_anomaly(True):

        for i in range(nb_epoch):
            if (i + 1) % (nb_epoch // 10) == 0:
                print(i + 1)
            # Log
            optimlog.zero_grad()
            losslog = multivariate_loglikelihood_torch((mulog, alphalog, betalog), tList)
            losslog.backward()
            optimlog.step()
            writer.add_scalar("LBFGS-Log", losslog, i)
            # Least
            optimlst.zero_grad()
            losslst = multivariate_lstsquares_torch((mulst, alphalst, betalst), tList)
            losslst.backward()
            optimlst.step()
            writer.add_scalar("LBFGS-Least", losslst, i)

        writer.close()

    print(torch.exp(mulog), alphalog, torch.exp(betalog))
    print(torch.exp(mulst), alphalst, torch.exp(betalst))

    os.system('tensorboard --logdir=' + path)
