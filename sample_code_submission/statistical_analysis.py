import numpy as np
from HiggsML.systematics import systematics

"""
Task 1a : Counting Estimator
1.write the saved_info dictionary such that it contains the following keys
    1. beta
    2. gamma
2. Estimate the mu using the formula
    mu = (sum(score * weight) - beta) / gamma
3. return the mu and its uncertainty

Task 1b : Stat-Only Likelihood Estimator
1. Modify the estimation of mu such that it uses the likelihood function
    1. Write a function for the likelihood function which profiles over mu
    2. Use Minuit to minimize the NLL

Task 2 : Systematic Uncertainty
1. substitute the beta and gamma with the tes_fit and jes_fit functions
2. Write a function to likelihood function which profiles over mu, tes and jes
3. Use Minuit to minimize the NLL
4. return the mu and its uncertainty

"""


def compute_mu(score, weight, saved_info):
    """
    Perform calculations to calculate mu
    Dummy code, replace with actual calculations
    Feel free to add more functions and change the function parameters

    """

    score = (score.ravel() > 0.5).astype(np.int8)

    '''
    np.sum(score * weight) = weighted count of events classified as Higgs.
    Subtract β (false-positive baseline) to remove expected background contamination.
    Divide by γ (true-positive baseline) to normalize so that μ̂ = 1 means “as many signal events as expected in reference”.
    
    Physics meaning: μ̂ ≈ measured signal yield / expected signal yield.:
        μ̂ = 1 → perfect agreement with reference model prediction.

        μ̂ > 1 → more Higgs-like events than expected.

        μ̂ < 1 → fewer Higgs-like events than expected.
    '''
    mu = (np.sum(score * weight) - saved_info["beta"]) / saved_info["gamma"]
    # Statistical uncertainty on μ̂.
    
    del_mu_stat = (
        np.sqrt(saved_info["beta"] + saved_info["gamma"]) / saved_info["gamma"]
    )
    del_mu_sys = abs(0.0 * mu)
    
    # Total uncertainty = quadrature sum of statistical and systematic uncertainties.
    del_mu_tot = np.sqrt(del_mu_stat**2 + del_mu_sys**2)

    return {
        "mu_hat": mu,
        "del_mu_stat": del_mu_stat,
        "del_mu_sys": del_mu_sys,
        "del_mu_tot": del_mu_tot,
    }


def calculate_saved_info(model, holdout_set):
    """
    Calculate the saved_info dictionary for mu calculation
    Replace with actual calculations
    """
    # These indicate how confident the model is that each event is signal (Higgs boson).
    score = model.predict(holdout_set["data"])
    # Type: np.ndarray, shape (N,) or (N,1)

    print("score shape before threshold", score.shape)

    '''
    flatten(): ensures it's 1D
    > 0.5: converts to boolean (True = Higgs, False = background)
    astype(int): converts True/False to 1/0
    Type: np.ndarray of int
    '''
    score = score.flatten() > 0.5
    score = score.astype(int)

    label = holdout_set["labels"] # Extract true labels

    print("score shape after threshold", score.shape)
    # Gamma = sum of weights for events predicted as signal AND truly are signal.
    # Measures how much true signal the model correctly captures. (Sum over all → total weighted true positives)
    # score * label: gives 1 only if prediction is 1 and true label is 1 → True Positive
    gamma = np.sum(holdout_set["weights"] * score * label)
    
    # Beta = sum of weights for events predicted as signal BUT are actually background.
    # Measures how much background the model incorrectly classifies as signal (false positives).
    # score * (1 - label): 1 only if predicted Higgs, but true label is background → False Positives
    beta = np.sum(holdout_set["weights"] * score * (1 - label))

    saved_info = {
        "beta": beta,
        "gamma": gamma,
    }

    print("saved_info", saved_info)

    return saved_info
