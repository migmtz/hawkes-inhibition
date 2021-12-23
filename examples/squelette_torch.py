import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from class_and_func.multivariate_exponential_process import multivariate_exponential_hawkes
from class_and_func.likelihood_functions import multivariate_loglikelihood_torch, multivariate_loglikelihood_simplified
import os


if __name__ == "__main__":
    ### Simulation of event times
    path = "/Users/alisdair/PycharmProjects/hawkes-inhibition/examples/runs/"
    torch.manual_seed(10)
    np.random.seed(10)

    dim = 2  # 2, 3 ou 4
    lr_log = 0.01
    lr_lst
    nb_epoch = 100

    mu = np.array([[0.5], [1.0]])
    alpha = np.array([[-0.9, 3], [1.2, 1.5]])
    beta = np.array([[4], [5]])

    max_jumps = 500

    hawkes = multivariate_exponential_hawkes(mu=mu, alpha=alpha, beta=beta, max_jumps=max_jumps)

    hawkes.simulate()

    tList = hawkes.timestamps

    mu = torch.nn.Parameter(torch.rand(dim, 1, dtype=torch.float64))
    alpha = torch.nn.Parameter(torch.randn(dim, dim, dtype=torch.float64))
    beta = torch.nn.Parameter(torch.rand(dim, 1, dtype=torch.float64))

    print(mu)
    print("torch", multivariate_loglikelihood_torch((mu, alpha, beta), tList))
    print("np", multivariate_loglikelihood_simplified((mu.detach().numpy(), alpha.detach().numpy(), beta.detach().numpy()), tList))

    parameters = [mu, alpha, beta]

    optiml = torch.optim.LBFGS(params=parameters, lr=lr_log)

    os.system('rm -rf ' + path)
    writer = SummaryWriter()

    for i in range(nb_epoch):
        if (i+1)%(nb_epoch//10):
            print(i+1)
        def closure():
            optiml.zero_grad()
            loss = multivariate_loglikelihood_torch((mu, alpha, beta), tList)
            loss.backward()
            writer.add_scalar("LBFGS", loss, i)
            return loss
        optiml.step(closure)

    writer.close()

    print(mu, alpha, torch.exp(beta))

    os.system('tensorboard --logdir=' + path)
