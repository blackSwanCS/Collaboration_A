import numpy as np


def calculate_mu(label, weight):
    """
    Perform calculations to calculate mu
    Dummy code, replace with actual calculations
    Feel free to add more functions and change the function parameters

    """
    mu = np.sum(label * weight)
    del_mu_stat = np.sqrt(np.sum(weight**2))
    del_mu_sys = 0.1 * mu
    del_mu_tot = np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }
