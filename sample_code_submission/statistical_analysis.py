import numpy as np
from HiggsML.systematics import systematics


def compute_mu(score, weight, saved_info):
    """
    Perform calculations to calculate mu
    Dummy code, replace with actual calculations
    Feel free to add more functions and change the function parameters

    """
    mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
    del_mu_stat = np.sqrt(saved_info["beta"] + saved_info["gamma"])/saved_info["gamma"]
    del_mu_sys = abs(0.1 * mu)
    del_mu_tot = (1 / 2) * np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }


def calculate_mu(model, train_data, train_labels, train_weight):
    """
    Dummy function to calculate mu
    Replace with actual calculations
    """
    train_set = {"data": train_data, "labels": train_labels, "weights": train_weight}

    train_plus_syst = systematics(
        data_set=train_set,
        tes=1.03,
        jes=1.03,
        soft_met=1.0,
        seed=31415,
        w_scale=None,
        bkg_scale=None,
        verbose=0,
    )

    train_minus_syst = systematics(
        data_set=train_set,
        tes=0.97,
        jes=0.97,
        soft_met=1.0,
        seed=31415,
        w_scale=None,
        bkg_scale=None,
        verbose=0,
    )

    score_plus_syst = model.predict(train_plus_syst["data"])
    score_minus_syst = model.predict(train_minus_syst["data"])

    print("Score plus syst: ", score_plus_syst.shape)
    print("Score minus syst: ", score_minus_syst.shape)
    print("Weights: ", train_set["weights"].shape)

    gamma = (
        np.sum(train_plus_syst["weights"] * score_plus_syst)
        + np.sum(train_minus_syst["weights"] * score_minus_syst)
    ) / 2
    beta = (
        np.sum(train_plus_syst["weights"] * (1 - score_plus_syst))
        + np.sum(train_minus_syst["weights"] * (1 - score_minus_syst))
    ) / 2

    saved_info = {"beta": beta, "gamma": gamma}

    return saved_info
