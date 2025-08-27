import numpy as np
from sys import path
import pickle
import matplotlib.pyplot as plt
from iminuit import Minuit
from checks import check_reweighting, check_calibration, roc_curve
import os
import pandas as pd
from iminuit import Minuit
import logging
from utils import (
    plot_calibration_curve,
    roc_curve_wrapper,
    plot_score_distributions,
    plot_two_score_distributions,
    plot_three_systematics_calibration,
    plot_score_distribution_for_triDataset,
)
import mlflow


path.append("../")
path.append("../ingestion_program")


from checks import plot_NLL

log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Write logs to file
        logging.StreamHandler(),  # Also print logs to console
    ],
)

logger = logging.getLogger(__name__)

# XGBOOST = True
XGBOOST = True

# TENSORFLOW = not XGBOOST
TENSORFLOW = False


current_dir = os.path.dirname(__file__)


class SystModel:
    """
    This class implements a model for the SBI (Simulation-Based Inference) submission.
    """

    def __init__(self, NP="tes", systematics=None, preselection=None, base_model=None):

        self.systematics = systematics
        self.alpha = [-1, 1]
        self.sigma = 0.01
        self.mean = 1
        self.re_train = True
        self.norm_syst = False
        self.lower_bound = 0.01
        self.upper_bound = 0.99
        self.NP = NP
        self.preselection = preselection
        self.systematics_values = [self.mean - self.sigma, self.mean + self.sigma]
        self.base_model = base_model
        self.columns = [
            "PRI_had_pt",
            "PRI_met",
            "PRI_met_phi",
            "DER_mass_transverse_met_lep",
            "DER_mass_vis",
            "DER_pt_h",
            "DER_deltar_had_lep",
            "DER_pt_tot",
            "DER_sum_pt",
            "DER_pt_ratio_lep_had",
            "DER_met_phi_centrality",
        ]

        # all columns
        self.base_features = [
            "PRI_lep_pt",
            "PRI_lep_eta",
            "PRI_lep_phi",
            "PRI_had_pt",
            "PRI_had_eta",
            "PRI_had_phi",
            "PRI_jet_leading_pt",
            "PRI_jet_leading_eta",
            "PRI_jet_leading_phi",
            "PRI_jet_subleading_pt",
            "PRI_jet_subleading_eta",
            "PRI_jet_subleading_phi",
            "PRI_n_jets",
            "PRI_jet_all_pt",
            "PRI_met",
            "PRI_met_phi",
            "DER_mass_transverse_met_lep",
            "DER_mass_vis",
            "DER_pt_h",
            "DER_deltaeta_jet_jet",
            "DER_mass_jet_jet",
            "DER_prodeta_jet_jet",
            "DER_deltar_had_lep",
            "DER_pt_tot",
            "DER_sum_pt",
            "DER_pt_ratio_lep_had",
            "DER_met_phi_centrality",
            "DER_lep_eta_centrality",
        ]

        if XGBOOST == True:
            from boosted_decision_tree import BoostedDecisionTree

            self.models = {
                "plus": BoostedDecisionTree(
                    name=f"{self.NP}_plus",
                    use_calibration=False,  # replaces .calibrate = True
                    calibration_method="isotonic",  # or "sigmoid" / "isotonic"
                    cv_calibration=False,  # optional: use cross-validated calibration
                    calibration_split=0.2,  # fraction of training data reserved for calibration
                ),
                "minus": BoostedDecisionTree(
                    name=f"{self.NP}_minus",
                    use_calibration=False,
                    calibration_method="isotonic",
                    cv_calibration=False,
                    calibration_split=0.2,
                ),
            }

            self.name = "BDT"

        elif TENSORFLOW == True:
            from neural_network import NeuralNetwork

            self.models = {
                "plus": NeuralNetwork(name=f"{self.NP}_plus"),
                "minus": NeuralNetwork(name=f"{self.NP}_minus"),
            }
            self.name = "NN"

        else:
            logger.error("No model selected")
            raise ValueError("No model selected")

        self.categories = ["ztautau", "ttbar", "diboson"]

        self.plots_dir = os.path.join(current_dir, "plots")
        self.model_dir = os.path.join(current_dir, "models")

        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # for key in self.models.keys():
        #     istrained = self.models[key].load(self.model_dir)
        self.istrained = False

    def density_ratio(self, score):
        logger.debug("score shape %s", score.shape)

        # Create a copy of the score array to avoid modifying the original
        score_new = score.copy()

        # Replace values greater than self.upper_bound with 0.99
        score_new[score > self.upper_bound] = self.upper_bound
        score_new[score < self.lower_bound] = self.lower_bound

        logger.debug("score_new shape %s", score_new.shape)

        return (1 - score_new) / (score_new)

    def train(self, dataset):

        self.model.fit(dataset["data"], dataset["labels"], dataset["weights"])

    """
    returns
    
    scores ={
        'plus': .... ,
        'minus': ....,
    }
    
    density_ratios ={
        'plus model': .... ,
        'minus model': ....,
    }
    """

    def predict_model(self, data):

        scores = {}
        density_ratios = {}
        for key in self.models.keys():

            model = self.models[key]
            if "base_model_score" not in data["data"].columns:
                data["data"]["base_model_score"] = self.base_model.predict(
                    data["data"][self.base_features]
                )
            score = model.predict(data["data"][self.columns])

            scores[key] = score
            density_ratios[key] = self.density_ratio(score)

        return scores, density_ratios

    def analyze(
        self,
        score,
        category="",
        training_set=None,
        holdout_set=None,
    ):

        calibration_pre2 = check_calibration(
            score,
            holdout_set["labels"],
            holdout_set["weights"],
            bins=50,
            filename=self.plots_dir + f"/{category}_calibration_pre2.png",
            label="",
        )
        run_dir = "mlruns_temp"
        os.makedirs(run_dir, exist_ok=True)
        mlflow.log_artifact(calibration_pre2)

        calib_path = plot_calibration_curve(
            data_den=training_set["data"]["score"].values[
                training_set["labels"].values == 0
            ],
            weight_den=training_set["weights"].values[
                training_set["labels"].values == 0
            ],
            data_num=training_set["data"]["score"].values[
                training_set["labels"].values == 1
            ],
            weight_num=training_set["weights"].values[
                training_set["labels"].values == 1
            ],
            data_denH=holdout_set["data"]["score"].values[
                holdout_set["labels"].values == 0
            ],
            weight_denH=holdout_set["weights"][holdout_set["labels"].values == 0],
            data_numH=holdout_set["data"]["score"].values[
                holdout_set["labels"].values == 1
            ],
            weight_numH=holdout_set["weights"][holdout_set["labels"].values == 1],
            epsilon=1.0e-20,
            label=f"Calibration Curve , {category} model",
            score_range="standard",
            save=f"{run_dir}/calibration_curve_{category}.pdf",
        )
        mlflow.log_artifact(calib_path)
        print(calib_path)
        """
        The density_ratio function computes a value for each event score that can be interpreted as a reweighting factor based on the score.
        It essentially transforms the model’s predicted score into a ratio that tells how to reweight signal events so that their distribution matches some reference 
        (e.g., background or systematic shifted distribution).
        """
        density_ratio = self.density_ratio(score)

        # ROC curve
        roc_path = roc_curve_wrapper(
            score=score,
            labels=holdout_set["labels"],
            weights=holdout_set["weights"],
            plot_label=f"Holdout_{category}",
            save_path=f"{run_dir}/roc_curve_{category}.png",
        )
        print(roc_path)
        mlflow.log_artifact(roc_path)

        # data["data"][f"score_{category}"] = score

        reweighting_paths = check_reweighting(density_ratio, holdout_set, category)
        for path in reweighting_paths:
            print(path)
            mlflow.log_artifact(path)
        print("score distribution graphs")

        score_dist_path = plot_score_distributions(
            training_set,
            holdout_set,
            category,
            self.preselection,
            self.columns,
            self.models,
            self.plots_dir,
            self.NP,
        )
        mlflow.log_artifact(score_dist_path)

    def fit(self, holdout_set, training_set=None, validation_set=None):
        """
        Trains the model.

        Params:
            None

        Functionality:
            This function can be used to train a model. If `re_train` is True, it balances the dataset,
            fits the model using the balanced dataset, and saves the model. If `re_train` is False, it
            loads the saved model and calculates the saved information. The saved information is used
            to compute the train results.

        Returns:
            None
        """
        holdout_data_sets = self.systematics_datasets(holdout_set)

        #  Gx is a weight ratio
        self.Gx = {}

        for key in holdout_data_sets.keys():

            # holdout_data_set = self.preselection.apply_pre_selection(
            #     holdout_data_sets[key],
            #     threshold=0.8,
            # )
            holdout_data_set = holdout_data_sets[key]

            # Gx = SUM OF SIGNALS weights / SUM OF BACKGROUNDS weights
            self.Gx[key] = np.sum(
                holdout_data_set["weights"][holdout_data_set["labels"] == 1]
            ) / np.sum(holdout_data_set["weights"][holdout_data_set["labels"] == 0])

            logger.debug("Gx %s", self.Gx[key])
            logger.debug("Gx shape %s", self.Gx[key].shape)
            print("Gx %s", self.Gx[key])
            print("Gx shape %s", self.Gx[key].shape)

        if self.norm_syst:
            return

        """
        If the model isn’t already trained:
        Build "plus" and "minus" training datasets.
        Balance event weights so the classifier sees equal effective class size
        """
        if not self.istrained:

            if training_set is None:
                logger.error("Training set not provided")
                raise ValueError("Training set not provided")
            """
            training_data_sets=
            {
                "plus": {
                    "data": <pandas.DataFrame>,    # features for nominal+shifted events (only plus variation)
                    "labels": <pandas.Series>,     # 0 = nominal, 1 = shifted
                    "weights": <pandas.Series>     # per-event training weights
                },
                "minus": {
                    "data": <pandas.DataFrame>,    # features for nominal+shifted events (only minus variation)
                    "labels": <pandas.Series>,     # 0 = nominal, 1 = shifted
                    "weights": <pandas.Series>     # per-event training weights
                }
            }

            """
            training_data_sets = self.systematics_datasets(training_set)
            # balancing weights same as in the model.py
            for key in training_data_sets.keys():  # key = "plus" or "minus"
                balanced_set = training_data_sets[key].copy()
                weights_train = training_data_sets[key]["weights"].copy()
                train_labels = training_data_sets[key]["labels"].copy()
                class_weights_train = (
                    weights_train[train_labels == 0].sum(),
                    weights_train[train_labels == 1].sum(),
                )
                for i in range(len(class_weights_train)):
                    weights_train[train_labels == i] *= (
                        max(class_weights_train) / class_weights_train[i]
                    )

                balanced_set["weights"] = weights_train

                print(balanced_set["data"].columns)

                print(len(weights_train[train_labels == 1]), f"label = 1 {key}")
                print(len(weights_train[train_labels == 0]), f"label = 0 {key}")

                self.models[key].fit(
                    training_data_sets[key]["data"],
                    training_data_sets[key]["labels"],
                    training_data_sets[key]["weights"],
                )

                # self.models[key].save(self.model_dir)
                del balanced_set

        # read and clean holdout set

        """
            holdout_data_sets=
            {
                "plus": {
                    "data": <pandas.DataFrame>,    # features for nominal+shifted events (only plus variation)
                    "labels": <pandas.Series>,     # 0 = nominal, 1 = shifted
                    "weights": <pandas.Series>     # per-event training weights
                },
                "minus": {
                    "data": <pandas.DataFrame>,    # features for nominal+shifted events (only minus variation)
                    "labels": <pandas.Series>,     # 0 = nominal, 1 = shifted
                    "weights": <pandas.Series>     # per-event training weights
                }
            }
            """

        for key in holdout_data_sets.keys():  # "plus" and "minus"

            holdout_score_temp = self.models[key].predict(
                holdout_data_sets[key]["data"],
            )
            holdout_data_sets[key]["data"]["score"] = holdout_score_temp
            training_data_sets[key]["data"]["score"] = self.models[key].predict(
                training_data_sets[key]["data"]
            )
            """
            So in this case of calibration curve --> , “num” and “den” are two variations of the dataset:
            For key="plus":
                num = +systematic variation
                den = nominal baseline
            For key="minus":
                num = −systematic variation
                den = nominal baseline
                
            calibration curve means : Given the classifier’s predicted score, 
            how often does the event really come from the shifted sample versus the nominal sample?
            
            and for ROC:
            X-axis: False Positive Rate (shifted events classified as nominal).
            Y-axis: True Positive Rate (shifted events correctly identified as shifted).

            for reweighting :
            
            This function is designed to check how well the model’s density ratio reweights the signal distribution to match the background distribution for each feature/variable in the dataset.


            """
            self.analyze(
                score=holdout_score_temp,
                category=key,
                training_set=training_data_sets[key],
                holdout_set=holdout_data_sets[key],
            )

            # test
            print(
                training_data_sets[key]["data"].shape,
                training_data_sets[key]["data"].columns,
            )
            print(
                holdout_data_sets[key]["data"].shape,
                holdout_data_sets[key]["data"].columns,
            )
            nominal_scores = holdout_data_sets[key]["data"]["score"][
                holdout_data_sets[key]["labels"] == 0
            ].values
            sys_scores = holdout_data_sets[key]["data"]["score"][
                holdout_data_sets[key]["labels"] == 1
            ].values
            ptsd = plot_two_score_distributions(
                nominal_scores,
                sys_scores,
                save_path=f"nominal&{key}-score_distribution.png",
                type=f"{key}",
            )
            mlflow.log_artifact(ptsd)

        pdrvagx = self.plot_density_ratio_vs_alpha_gx(holdout_set, save_path=None)
        mlflow.log_artifact(pdrvagx)
        pdrsvagx = self.plot_density_ratio_scatter_vs_alpha_gx(
            holdout_set, save_path=None
        )
        mlflow.log_artifact(pdrsvagx)

        holdout_set_norm = self.systematics(holdout_set)

        preselected_holdout = self.preselection.apply_pre_selection(
            holdout_set_norm,
            threshold=0.8,
        )

        _, holdout_density_ratios = self.predict_model(preselected_holdout)

        self.fit_extropolate(holdout_density_ratios)

        pdva = self.plot_pdf_vs_alpha(holdout_set)
        mlflow.log_artifact(pdva)
        pllva = self.plot_log_likelihood_vs_alpha(holdout_set)
        mlflow.log_artifact(pllva)

        tri_holdout = self.systematics_datasets2(holdout_set)
        for key in self.models.keys():
            tri_scores = self.models[key].predict(tri_holdout["data"])
            scores_nominal = tri_scores[tri_holdout["labels"] == 0]
            scores_plus = tri_scores[tri_holdout["labels"] == 1]
            scores_minus = tri_scores[tri_holdout["labels"] == -1]

            tri_holdout_df = tri_holdout["data"].copy()
            tri_holdout_df["label"] = tri_holdout["labels"].values
            tri_holdout_df["weight"] = tri_holdout["weights"].values
            tri_holdout_df[f"score_{key}Model"] = tri_scores

            psdftd = plot_score_distribution_for_triDataset(
                scores_nominal,
                scores_plus,
                scores_minus,
                save_path=f"nominal&Plus&minus-score_distribution_for_{key}-Model.png",
                type=f"{key}",
            )
            mlflow.log_artifact(psdftd)

        # ptsc = plot_three_systematics_calibration(
        #     nominal_scores=holdout_data_sets["plus"]["data"]["score"].values,
        #     nominal_labels=holdout_data_sets["plus"]["labels"].values,
        #     nominal_weights=holdout_data_sets["plus"]["weights"].values,
        #     plus_scores=holdout_data_sets["plus"]["data"]["score"].values,
        #     plus_labels=holdout_data_sets["plus"]["labels"].values,
        #     plus_weights=holdout_data_sets["plus"]["weights"].values,
        #     minus_scores=holdout_data_sets["minus"]["data"]["score"].values,
        #     minus_labels=holdout_data_sets["minus"]["labels"].values,
        #     minus_weights=holdout_data_sets["minus"]["weights"].values,
        #     bins=50,
        #     save_path=f"{current_dir}/plots/systematics_score_comparison.png",
        # )
        # mlflow.log_artifact(ptsc)

        holdout_set_norm = self.systematics(holdout_set)

        preselected_holdout = self.preselection.apply_pre_selection(
            holdout_set_norm,
            threshold=0.8,
        )

        _, holdout_density_ratios = self.predict_model(preselected_holdout)

        self.fit_extropolate(holdout_density_ratios)

        Gx_0_holdout = self.syst_fun(self.coeff_G, 1)
        gx_0_holdout = self.syst_fun(self.coeff_g, 1)

        Gx_plus = self.Gx["plus"]
        Gx_minus = self.Gx["minus"]

        logger.debug("Gx_plus %s", Gx_plus)
        logger.debug("Gx_minus %s", Gx_minus)
        print("Gx_plus %s", Gx_plus)
        print("Gx_minus %s", Gx_minus)

        gx_plus = self.predict(preselected_holdout)["plus"]
        gx_minus = self.predict(preselected_holdout)["minus"]

        logger.debug("Gx_0_holdout %s", Gx_0_holdout)
        logger.debug("gx_0_holdout %s", gx_0_holdout)

        pdf_holdout = self.pdf(1, gx_plus, gx_minus)
        pdf_holdout_plus = self.pdf(1.01, gx_plus, gx_minus)
        pdf_holdout_minus = self.pdf(0.99, gx_plus, gx_minus)

        logger.debug("pdf_holdout %s", pdf_holdout)
        logger.debug("pdf_holdout_plus %s", pdf_holdout_plus)
        logger.debug("pdf_holdout_minus %s", pdf_holdout_minus)

        # simple quadratic

    def syst_fun(self, coeff, alpha):
        return 1 + coeff[0] * alpha + coeff[1] * alpha**2

    def Wrong_plot_density_ratio_vs_alpha_gx(self, dataset, save_path=None):
        data_sets = self.systematics_datasets(dataset)
        scores_plus, density_ratios_plus = self.predict_model(data_sets["plus"])
        scores_minus, density_ratios_minus = self.predict_model(data_sets["minus"])
        self.fit_extropolate(density_ratios_plus)  # fit for "plus" data
        coeff_g_plus = self.coeff_g
        coeff_G_plus = self.coeff_G

        self.fit_extropolate(density_ratios_minus)  # fit for "minus" data
        coeff_g_minus = self.coeff_g
        coeff_G_minus = self.coeff_G

        alphas = np.linspace(0.9, 1.1, 50)
        density_plus_curve = np.array([self.syst_fun(coeff_g_plus, a) for a in alphas])
        density_minus_curve = np.array(
            [self.syst_fun(coeff_g_minus, a) for a in alphas]
        )

        plt.figure(figsize=(8, 5))
        plt.plot(alphas, density_plus_curve, label="Plus variation", color="blue")
        plt.plot(alphas, density_minus_curve, label="Minus variation", color="red")
        plt.xlabel("Alpha")
        plt.ylabel("Density ratio")
        plt.title("Density Ratio vs Alpha")
        plt.legend()
        plt.grid(True)
        run_dir = "mlruns_temp"
        os.makedirs(run_dir, exist_ok=True)
        save_path = f"{run_dir}/Wrong_ratio_vs_alpha_graph.png"
        plt.savefig(save_path)
        plt.show()
        plt.close()
        return save_path

    def plot_density_ratio_vs_alpha_gx(self, dataset, save_path=None):
        # Prepare alpha values from systematics range
        alpha_min, alpha_max = self.systematics_values
        alphas = np.linspace(alpha_min, alpha_max, 10)

        # Store results
        density_ratios_plus = []
        density_ratios_minus = []
        syst_fixed_setting = {
            "tes": 1.0,
            "bkg_scale": 1.0,
            "jes": 1.0,
            "soft_met": 0.0,
            "ttbar_scale": 1.0,
            "diboson_scale": 1.0,
        }

        syst_setting = syst_fixed_setting.copy()
        if self.NP in syst_setting.keys():
            for alpha in alphas:
                syst_setting[self.NP] = alpha
                # Apply systematic variation
                dataset_syst = self.systematics(
                    dataset.copy(), dopostprocess=True, **syst_setting
                )
                # Run predictions
                scores, density_ratios = self.predict_model(dataset_syst)
                # Collect the mean density ratio for each model
                # (assuming you want an aggregate; otherwise keep the whole array)
                weights = dataset_syst["weights"].values
                density_ratios_plus.append(
                    np.average(density_ratios["plus"], weights=weights)
                )
                density_ratios_minus.append(
                    np.average(density_ratios["minus"], weights=weights)
                )

            # --- Plotting ---
            plt.figure(figsize=(8, 5))
            plt.plot(
                alphas,
                density_ratios_plus,
                label="Plus model",
                color="blue",
                marker="o",
            )
            plt.plot(
                alphas,
                density_ratios_minus,
                label="Minus model",
                color="red",
                marker="o",
            )
            plt.xlabel("Alpha")
            plt.ylabel("Density ratio (mean)")
            plt.title(f"Density Ratios vs {self.NP}")
            plt.legend()
            plt.grid(True)
            run_dir = "mlruns_temp"
            os.makedirs(run_dir, exist_ok=True)
            save_path = f"{run_dir}/Densityratio_vs_alpha_graph.png"
            plt.savefig(save_path)
            plt.show()
            plt.close()
            return save_path

    def plot_density_ratio_scatter_vs_alpha_gx(self, dataset, save_path=None):
        # Prepare alpha values from systematics range
        alpha_min, alpha_max = self.systematics_values
        alphas = np.linspace(alpha_min, alpha_max, 10)

        # Store results
        density_ratios_plus = []
        density_ratios_minus = []
        syst_fixed_setting = {
            "tes": 1.0,
            "bkg_scale": 1.0,
            "jes": 1.0,
            "soft_met": 0.0,
            "ttbar_scale": 1.0,
            "diboson_scale": 1.0,
        }

        syst_setting = syst_fixed_setting.copy()
        if self.NP in syst_setting.keys():
            all_alphas_plus = []
            all_ratios_plus = []
            all_alphas_minus = []
            all_ratios_minus = []

            for alpha in alphas:
                syst_setting[self.NP] = alpha
                # Apply systematic variation
                dataset_syst = self.systematics(
                    dataset.copy(), dopostprocess=True, **syst_setting
                )
                scores, density_ratios = self.predict_model(dataset_syst)

                # all density ratios so not just the mean for scatter
                all_alphas_plus.extend([alpha] * len(density_ratios["plus"]))
                all_ratios_plus.extend(density_ratios["plus"])

                all_alphas_minus.extend([alpha] * len(density_ratios["minus"]))
                all_ratios_minus.extend(density_ratios["minus"])

            # --- Plotting scatter ---
            plt.figure(figsize=(8, 5))
            plt.scatter(
                all_alphas_plus,
                all_ratios_plus,
                label="Plus model",
                color="blue",
                alpha=0.5,
                s=10,
            )
            plt.scatter(
                all_alphas_minus,
                all_ratios_minus,
                label="Minus model",
                color="red",
                alpha=0.5,
                s=10,
            )

            plt.xlabel("Alpha")
            plt.ylabel("Density ratio")
            plt.title(f"Scatter of Density Ratios vs {self.NP}")
            plt.legend()
            plt.grid(True)

            run_dir = "mlruns_temp"
            os.makedirs(run_dir, exist_ok=True)
            save_path = f"{run_dir}/Densityratio_vs_alpha_scatter.png"
            plt.savefig(save_path)
            plt.show()
            plt.close()
            return save_path

    def plot_pdf_vs_alpha(self, dataset, save_path=None):
        """
        Plot the PDF function across alpha values - this shows your likelihood function!
        """
        # Use your existing holdout predictions
        preselected_holdout = self.preselection.apply_pre_selection(
            dataset, threshold=0.8
        )
        gx_predictions = self.predict(preselected_holdout)
        gx_plus = gx_predictions["plus"]
        gx_minus = gx_predictions["minus"]

        # Alpha range including the extrapolation regions
        alphas = np.linspace(0.99, 1.01, 100)

        # Calculate PDF for each alpha - this is your likelihood function!
        pdf_values = []
        pdf_mean_values = []

        for alpha in alphas:
            pdf_alpha = self.pdf(alpha, gx_plus, gx_minus)
            pdf_values.append(pdf_alpha)
            # Store mean for cleaner plotting
            pdf_mean_values.append(np.mean(pdf_alpha))

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(alphas, pdf_mean_values, "b-", linewidth=2, label="PDF(α)")
        ax1.axvline(x=1.0, color="k", linestyle="--", alpha=0.7, label="Nominal")
        ax1.axvline(x=0.99, color="r", linestyle=":", alpha=0.7, label="Minus anchor")
        ax1.axvline(x=1.01, color="g", linestyle=":", alpha=0.7, label="Plus anchor")

        # ax1.axvspan(
        #     0.99, self.alpha[0], alpha=0.1, color="red", label="Minus extrapolation"
        # )
        # ax1.axvspan(
        #     self.alpha[0],
        #     self.alpha[1],
        #     alpha=0.1,
        #     color="yellow",
        #     label="Interpolation",
        # )
        # ax1.axvspan(
        #     self.alpha[1], 1.01, alpha=0.1, color="blue", label="Plus extrapolation"
        # )

        ax1.set_xlabel("Alpha (Systematic Parameter)")
        ax1.set_ylabel("PDF Value (Mean)")
        ax1.set_title("Likelihood Function: PDF vs Alpha")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom plot: PDF distribution at key alpha values
        key_alphas = [0.99, 1.0, 1.01]
        colors = ["red", "black", "green"]
        labels = ["α=0.99 (Minus)", "α=1.0 (Nominal)", "α=1.01 (Plus)"]

        for i, alpha in enumerate(key_alphas):
            pdf_alpha = self.pdf(alpha, gx_plus, gx_minus)
            # Plot histogram of PDF values
            ax2.hist(
                pdf_alpha,
                bins=50,
                alpha=0.6,
                color=colors[i],
                label=labels[i],
                density=True,
            )

        ax2.set_xlabel("PDF Value")
        ax2.set_ylabel("Density")
        ax2.set_title("PDF Distributions at Key Alpha Values")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        run_dir = "mlruns_temp"
        os.makedirs(run_dir, exist_ok=True)
        save_path = f"{run_dir}/pdf_vs_alpha.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

        return save_path

    def plot_log_likelihood_vs_alpha(self, dataset, save_path=None):
        """
        Plot log-likelihood vs alpha - often more interpretable
        """
        preselected_holdout = self.preselection.apply_pre_selection(
            dataset, threshold=0.8
        )
        gx_predictions = self.predict(preselected_holdout)
        gx_plus = gx_predictions["plus"]
        gx_minus = gx_predictions["minus"]

        alphas = np.linspace(0.95, 1.05, 100)
        log_likelihood_values = []

        for alpha in alphas:
            pdf_alpha = self.pdf(alpha, gx_plus, gx_minus)
            # Sum log-likelihood (assuming independent events)
            log_likelihood = np.sum(
                np.log(pdf_alpha + 1e-10)
            )  # Small epsilon for numerical stability
            log_likelihood_values.append(log_likelihood)

        plt.figure(figsize=(10, 6))
        plt.plot(
            alphas, log_likelihood_values, "b-", linewidth=2, label="Log-Likelihood"
        )
        plt.axvline(x=1.0, color="k", linestyle="--", alpha=0.7, label="Nominal")
        plt.axvline(x=0.99, color="r", linestyle=":", alpha=0.7, label="Minus anchor")
        plt.axvline(x=1.01, color="g", linestyle=":", alpha=0.7, label="Plus anchor")

        # Find and mark the maximum
        max_idx = np.argmax(log_likelihood_values)
        max_alpha = alphas[max_idx]
        max_ll = log_likelihood_values[max_idx]
        plt.scatter(
            [max_alpha],
            [max_ll],
            color="red",
            s=100,
            zorder=5,
            label=f"Maximum at α={max_alpha:.3f}",
        )

        plt.xlabel("Alpha (Systematic Parameter)")
        plt.ylabel("Log-Likelihood")
        plt.title("Log-Likelihood Function vs Alpha")
        plt.legend()
        plt.grid(True, alpha=0.3)

        run_dir = "mlruns_temp"
        os.makedirs(run_dir, exist_ok=True)
        save_path = f"{run_dir}/log_likelihood_vs_alpha.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

        return save_path

    def fit_extropolate(self, density_ratios):

        # creates a 3d polynomial fit for the density ratios, which depend on alpha
        # the fit is used to extrapolate the density ratios to alpha = 1 + 0.01 and alpha = 1 - 0.01
        """
        self.syst_fun(coeff, α_plus) is a single scalar
        Subtracting that scalar from a vector (e.g. density_ratios["plus"]) creates a residual vector;
        squaring and summing gives a standard sum of squared errors (SSE).
        """

        # cost= (error at plus)^2 + (error at minus)^2
        # Density Ratio Fitting
        def cost_g(coeff_0, coeff_1):
            coeff = [coeff_0, coeff_1]
            return np.sum(
                # (model_prediction - expected_value)²
                (self.syst_fun(coeff, 1) - density_ratios["plus"]) ** 2
                + (self.syst_fun(coeff, -1) - density_ratios["minus"]) ** 2
                + (self.syst_fun(coeff, 0) - 1) ** 2
            )

        # Create an optimizer to minimize the cost function cost_g starting from 0,0.
        m = Minuit(cost_g, coeff_0=0, coeff_1=0)
        # Run the optimizer to find the values of coeff_0 and coeff_1 that make cost_g as small as possible.
        m.migrad()
        # After this, self.coeff_Gg contains the fitted coefficients for your quadratic systematic model.
        self.coeff_g = [m.values["coeff_0"], m.values["coeff_1"]]

        # Gx = SUM OF SIGNALS weights / SUM OF BACKGROUNDS weights
        def cost_G(coeff_0, coeff_1):
            coeff = [coeff_0, coeff_1]
            return np.sum(
                (self.syst_fun(coeff, 1) - self.Gx["plus"]) ** 2
                + (self.syst_fun(coeff, -1) - self.Gx["minus"]) ** 2
                + (self.syst_fun(coeff, 0) - 1) ** 2
            )

        """
        It’s a numerical minimizer designed for scientific problems, widely used in physics.
        Minuit finds the parameter values that minimize it.
        
        cost_G is the function to minimize (in your case, the sum of squared differences between the quadratic model and observed data at the plus/minus anchors).

        coeff_0=0, coeff_1=0 are the initial guesses for the parameters coeff_0 and coeff_1.

        Think of this as telling Minuit: “Start here (0,0) and try to find the best coeff_0 and coeff_1 that make cost_G as small as possible.”
        """
        m = Minuit(cost_G, coeff_0=0, coeff_1=0)

        """
        runs the MIGRAD algorithm, which is:

        A quasi-Newton optimizer.

        Uses gradients (or approximates them numerically) to iteratively find the minimum of the cost function.
        """
        m.migrad()
        """
        m.values contains the optimized parameters found by MIGRAD.
        So now self.coeff_G stores the best-fit coefficients for your quadratic model
        """
        self.coeff_G = [m.values["coeff_0"], m.values["coeff_1"]]

    def pdf(self, alpha, g_x_plus, g_x_minus):
        if alpha > self.alpha[1]:
            pdf = (self.Gx["plus"] * g_x_plus) ** alpha
        elif alpha < self.alpha[0]:
            pdf = (self.Gx["minus"] * g_x_minus) ** (-alpha)
        else:
            pdf = (
                self.syst_fun(self.coeff_g, alpha)
                * self.syst_fun(self.coeff_G, alpha)
                * np.ones(len(g_x_plus))
            )

        logger.debug("pdf shape %s", pdf.shape)

        return pdf

    # returns DR
    def predict(self, data):
        """
        Predicts the class of the data.

        Args:
            data (dict): A dictionary containing the data.

        Returns:
            dict: A dictionary containing the predicted class of the data.
        """
        data.pop("score", None)
        # data = data[self.columns]
        if "base_model_score" not in data["data"].columns:
            data["data"]["base_model_score"] = self.base_model.predict(
                data["data"][self.base_features]
            )

        data["data"] = data["data"][self.columns]
        _, density_ratios = self.predict_model(data)

        return density_ratios

    def systematics_datasets(self, dataset):
        """
        Add systematics to the dataset.

        Args:
            dataset (dict): A dictionary containing the dataset.

        Returns:
            dict: A dictionary containing the dataset with systematics.
        """

        syst_fixed_setting = {
            "tes": 1.0,
            "bkg_scale": 1.0,
            "jes": 1.0,
            "soft_met": 0.0,
            "ttbar_scale": 1.0,
            "diboson_scale": 1.0,
        }
        indiviual_datasets = {}

        syst_setting = syst_fixed_setting.copy()

        syst_setting[self.NP] = 1.0
        dataset_nom = self.systematics(
            dataset.copy(), dopostprocess=True, **syst_fixed_setting
        )

        # Assigns label 0 to all nominal events (important for classification).
        df_nom = dataset_nom["data"]
        df_nom["labels"] = np.zeros(len(df_nom))  # Nominal = class 0
        df_nom["weights"] = dataset_nom["weights"]  # <-- Add this line
        indiviual_datasets["nominal"] = df_nom

        print(df_nom["weights"].sum(), "sum of the weights of df nominal")
        print("length of df of the nominal", len(df_nom))

        del dataset_nom

        data_sets = {}
        if self.NP in syst_setting.keys():

            for syst_value in self.systematics_values:  # [0.99, 1.01]
                syst_setting[self.NP] = syst_value

                name = "minus" if syst_value < syst_fixed_setting[self.NP] else "plus"
                print("syst_value", syst_value)
                print(self.NP, " :Hello this is NP")
                print("this is the shifted for the ", name)
                print(syst_fixed_setting, "syst_fixed_setting")
                print(syst_setting, "syst_setting")

                dataset_syst = self.systematics(
                    dataset.copy(), dopostprocess=True, **syst_setting
                )
                df_syst = dataset_syst["data"]

                df_syst["labels"] = np.ones(len(df_syst))  # Shifted = class 1

                df_syst["weights"] = dataset_syst["weights"]

                print(df_syst["weights"].sum(), "sum of the weights of df sys")
                print(df_nom["weights"].sum(), "sum of the weights of df nom")

                print(df_syst["weights"].count(), "length of the weights of df sys")
                print(df_nom["weights"].count(), "length of the weights of df nom")

                # Merge nominal + shifted for binary classification
                df = pd.concat([df_nom, df_syst]).reset_index(drop=True)

                df = df.sample(frac=1)  # shuffle rows

                labels = df.pop("labels")
                # df.pop("score")
                base_features = self.base_features
                df["base_model_score"] = self.base_model.predict(df[base_features])
                weights = df.pop("weights")
                if "base_model_score" not in self.columns:
                    self.columns.append("base_model_score")
                df = df[self.columns]
                """

                """

                data_sets[f"{name}"] = {
                    "data": df,
                    "labels": labels,
                    "weights": weights,
                }
                del df

        else:
            raise ValueError("Systematics not implemented")

        return data_sets

    def systematics_datasets2(self, dataset):
        """
        Build a dataset with three categories: minus (-1), nominal (0), plus (+1).

        Args:
            dataset (dict): A dictionary containing the dataset.

        Returns:
            dict: A dictionary containing one dataset with 'data', 'labels', 'weights'.
        """
        # Fixed settings
        syst_fixed_setting = {
            "tes": 1.0,
            "bkg_scale": 1.0,
            "jes": 1.0,
            "soft_met": 0.0,
            "ttbar_scale": 1.0,
            "diboson_scale": 1.0,
        }
        syst_setting = syst_fixed_setting.copy()

        # Nominal dataset (label = 0)
        dataset_nom = self.systematics(
            dataset.copy(), dopostprocess=True, **syst_fixed_setting
        )
        df_nom = dataset_nom["data"]
        df_nom["labels"] = np.zeros(len(df_nom))
        df_nom["weights"] = dataset_nom["weights"]

        all_dfs = [df_nom]

        del dataset_nom
        # Loop over systematics values ([1.1, 0.9])
        for syst_value in self.systematics_values:
            syst_setting[self.NP] = syst_value

            # Decide if it's plus or minus
            label = 1 if syst_value > syst_fixed_setting[self.NP] else -1
            name = "plus" if label == 1 else "minus"

            dataset_syst = self.systematics(
                dataset.copy(), dopostprocess=True, **syst_setting
            )
            df_syst = dataset_syst["data"]
            df_syst["labels"] = label
            df_syst["weights"] = dataset_syst["weights"]

            all_dfs.append(df_syst)

        df = pd.concat(all_dfs).sample(frac=1).reset_index(drop=True)

        labels = df.pop("labels")
        # df.pop("score")
        base_features = self.base_features
        df["base_model_score"] = self.base_model.predict(df[base_features])
        weights = df.pop("weights")
        if "base_model_score" not in self.columns:
            self.columns.append("base_model_score")
        df = df[self.columns]

        return {
            "data": df,
            "labels": labels,
            "weights": weights,
        }

    def save(self):

        # saving coefficients
        with open(self.model_dir + "/coeff_g.pkl", "wb") as f:
            pickle.dump(self.coeff_g, f)

        with open(self.model_dir + "/coeff_G.pkl", "wb") as f:
            pickle.dump(self.coeff_G, f)

        # saving Gx
        with open(self.model_dir + "/Gx.pkl", "wb") as f:
            pickle.dump(self.Gx, f)

        logger.info("Model saved")

    def load(self):

        with open(self.model_dir + "/coeff_g.pkl", "rb") as f:
            self.coeff_g = pickle.load(f)

        with open(self.model_dir + "/coeff_G.pkl", "rb") as f:
            self.coeff_G = pickle.load(f)

        with open(self.model_dir + "/Gx.pkl", "rb") as f:
            self.Gx = pickle.load(f)

        logger.info("Model loaded")
