# ------------------------------
# Dummy Sample Submission
# ------------------------------

BDT = False
NN = True

from statistical_analysis import calculate_mu, compute_mu
from feature_engineering import feature_engineering
from derived_features import derived_feature
from HiggsML.datasets import train_test_split
import HiggsML.visualization as visualization

class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init :
        takes 2 arguments: train_set and systematics,
        can be used for intiializing variables, classifier etc.
    2) fit :
        takes no arguments
        can be used to train a classifier
    3) predict:
        takes 1 argument: test sets
        can be used to get predictions of the test set.
        returns a dictionary

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods

            When you add another file with the submission model e.g. a trained model to be loaded and used,
            load it in the following way:

            # get to the model directory (your submission directory)
            model_dir = os.path.dirname(os.path.abspath(__file__))

            your trained model file is now in model_dir, you can load it from here
    """

    def __init__(self, get_train_set=None, systematics=None):
        """
        Model class constructor

        Params:
            train_set:
                a dictionary with data, labels, weights and settings

            systematics:
                a class which you can use to get a dataset with systematics added
                See sample submission for usage of systematics


        Returns:
            None
        """
        self.train_set = (
            get_train_set  # train_set is a dictionary with data, labels and weights
        )
        self.systematics = systematics
        
        del self.train_set["settings"]

        
        print("Full data: ", self.train_set["data"].shape)
        print("Full Labels: ", self.train_set["labels"].shape)
        print("Full Weights: ", self.train_set["weights"].shape)
        print("sum_signal_weights: ", self.train_set["weights"][self.train_set["labels"] == 1].sum())
        print("sum_bkg_weights: ", self.train_set["weights"][self.train_set["labels"] == 0].sum())
        print(" \n ")
        
        
        self.training_set , self.valid_set = train_test_split(data_set = self.train_set, test_size=0.2, random_state=42, reweight=True)

        del  self.train_set
        
        print("Training Data: ", self.training_set["data"].shape)
        print("Training Labels: ", self.training_set["labels"].shape)
        print("Training Weights: ", self.training_set["weights"].shape)
        print("sum_signal_weights: ", self.training_set["weights"][self.training_set["labels"] == 1].sum())
        print("sum_bkg_weights: ", self.training_set["weights"][self.training_set["labels"] == 0].sum())
        print()
        print("Valid Data: ", self.valid_set["data"].shape)
        print("Valid Labels: ", self.valid_set["labels"].shape)
        print("Valid Weights: ", self.valid_set["weights"].shape)
        print("sum_signal_weights: ", self.valid_set["weights"][self.valid_set["labels"] == 1].sum())
        print("sum_bkg_weights: ", self.valid_set["weights"][self.valid_set["labels"] == 0].sum())
        print(" \n ")

        train_data_with_derived_features = derived_feature(self.training_set["data"])

        self.training_set["data"] = feature_engineering(train_data_with_derived_features)

        print("Training Data: ", self.training_set["data"].shape)

        if BDT:
            from boosted_decision_tree import BoostedDecisionTree

            self.model = BoostedDecisionTree(train_data=self.training_set["data"])

            print("Model is BDT")
        else:
            from neural_network import NeuralNetwork

            self.model = NeuralNetwork(train_data=self.training_set["data"])

            print("Model is NN")

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model

        Returns:
            None
        """

        self.model.fit(self.training_set["data"], self.training_set["labels"])

        self.saved_info = calculate_mu(
            self.model,  self.training_set
        )

        train_score = self.model.predict(self.training_set["data"])
        train_results = compute_mu(train_score, self.training_set["weights"], self.saved_info)

        valid_data_with_derived_features = derived_feature(self.valid_set["data"])

        self.valid_set["data"] = feature_engineering(valid_data_with_derived_features)

        valid_score = self.model.predict(self.valid_set["data"])

        valid_results = compute_mu(valid_score, self.valid_set["weights"], self.saved_info)

        print("train Results: ", train_results)
        print("valid Results: ", valid_results)
        
        self.valid_set["data"]["score"] = valid_score
        

        
        valid_visualize = visualization.Dataset_visualise(
            data_set=self.valid_set,
            columns=[
                "PRI_jet_leading_pt",
                "PRI_met",
                "score",

            ],
            name="Train Set",
        )
        
        valid_visualize.histogram_dataset()
        valid_visualize.stacked_histogram("score")

    def predict(self, test_set):
        """
        Params:
            test_set

        Functionality:
            this function can be used for predictions using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """

        test_data = derived_feature(test_set["data"])
        test_data = feature_engineering(test_data)
        test_weights = test_set["weights"]

        predictions = self.model.predict(test_data)

        result_mu_cal = compute_mu(predictions, test_weights, self.saved_info)

        print("Test Results: ", result_mu_cal)

        result = {
            "mu_hat": result_mu_cal["mu_hat"],
            "delta_mu_hat": result_mu_cal["del_mu_tot"],
            "p16": result_mu_cal["mu_hat"] - result_mu_cal["del_mu_tot"],
            "p84": result_mu_cal["mu_hat"] + result_mu_cal["del_mu_tot"],
        }

        return result
