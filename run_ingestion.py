import sys

sys.path.append("..")
from HiggsML.ingestion import Ingestion

from HiggsML.datasets import Data
import argparse
import pathlib
import os
import numpy as np
import json

root_dir_name = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    description="This is script to run ingestion program for the competition"
)
parser.add_argument(
    "--input",
    "-i",
    type=pathlib.Path,
    help="Input file location",
    default=os.path.join(root_dir_name, "sample_data"),
)
parser.add_argument(
    "--output",
    "-o",
    help="Output file location",
    default=os.path.join(root_dir_name, "sample_result_submission"),
)
parser.add_argument(
    "--submission",
    "-s",
    help="Submission file location",
    default=os.path.join(root_dir_name, "sample_code_submission"),
)
parser.add_argument(
    "--codabench",
    help="True when running on Codabench",
    action="store_true",
)
parser.add_argument(
    "--use-random-mus",
    help="Use random mus for testing",
    action="store_true",
)
parser.add_argument(
    "--systematics-tes",
    action="store_true",
    help="Whether to use tes systematics",
)
parser.add_argument(
    "--systematics-jes",
    action="store_true",
    help="Whether to use jes systematics",
)
parser.add_argument(
    "--systematics-soft-met",
    action="store_true",
    help="Whether to use soft_met systematics",
)
parser.add_argument(
    "--systematics-w-scale",
    action="store_true",
    help="Whether to use w_scale systematics",
)
parser.add_argument(
    "--systematics-bkg-scale",
    action="store_true",
    help="Whether to use bkg_scale systematics",
)
parser.add_argument(
    "--num-pseudo-experiments",
    type=int,
    help="Number of pseudo experiments",
    default=2,
)
parser.add_argument(
    "--num-of-sets",
    type=int,
    help="Number of sets",
    default=5,
)


args = parser.parse_args()

if not args.codabench:
    input_dir = args.input
    output_dir = args.output
    submission_dir = args.submission
else:
    input_dir = "/app/input_data"
    output_dir = "/app/output"
    submission_dir = "/app/ingested_program"
    program_dir = "/app/program"


if not args.codabench:
    from HiggsML.datasets import BlackSwan_public_dataset as public_dataset

    data = public_dataset()
else:
    data = Data(input_dir,data_format="parquet")


sys.path.append(submission_dir)

from model import Model


ingestion = Ingestion(data)

# Start timer
ingestion.start_timer()

# initialize submission
ingestion.init_submission(Model)

# fit submission
ingestion.fit_submission()
test_settings = {}
test_settings["systematics"] = {
    "tes": args.systematics_tes,
    "jes": args.systematics_jes,
    "soft_met": args.systematics_soft_met,
    "w_scale": args.systematics_w_scale,
    "bkg_scale": args.systematics_bkg_scale,
}

test_settings["num_pseudo_experiments"] = args.num_pseudo_experiments
test_settings["num_of_sets"] = args.num_of_sets

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if args.use_random_mus:
    test_settings["ground_truth_mus"] = (
        np.random.uniform(0.1, 3, test_settings["num_of_sets"])
    ).tolist()
    test_settings["random_mu"] = True
    random_settings_file = os.path.join(output_dir, "test_settings.json")
    with open(random_settings_file, "w") as f:
        json.dump(test_settings, f)
else:
    test_settings["ground_truth_mus"] = data.ground_truth_mus

# load test data
data.load_test_set()

# predict submission
ingestion.predict_submission(test_settings)

# compute result
ingestion.process_results_dict()

# save result
ingestion.save_result(output_dir)

# Stop timer
ingestion.stop_timer()

# Show duration
ingestion.show_duration()

# Save duration
ingestion.save_duration(output_dir)
