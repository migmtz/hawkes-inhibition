import numpy as np


def relative_squared_loss(theta, est):
    theta_low = theta.copy()
    dim = int(np.sqrt(1 +theta.shape[0]) - 1)
    theta_low[-dim:][theta_low[-dim:] == 0] = est[-dim:][theta_low[-dim:] == 0]
    theta_low[theta_low == 0] = 1
    theta_low = theta_low**2

    num = (theta - est)**2

    mu_error = np.sqrt(np.sum(num[0:dim])/np.sum(theta_low[0:dim]))
    alpha_error = np.sqrt(np.sum(num[dim:dim+dim*dim]) / np.sum(theta_low[dim:dim+dim*dim]))
    beta_error = np.sqrt(np.sum(num[-dim:]) / np.sum(theta_low[-dim:]))

    full_error = np.sqrt(np.sum(num)/np.sum(theta_low))

    return mu_error, alpha_error, beta_error, full_error


if __name__ == "__main__":
    a = np.array([1,2,3,4,5,6,0,8])
    b = np.array([1,3,3,4,5,7,10,12])

    print(relative_squared_loss(a,b))