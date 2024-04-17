############################
#OI HS Analysis Script     #
#Maintainer: Timur Burkanov#
#Date: 2024-04-17          #
#Version: 0.0.1            #
############################

import os
import pathlib
import matplotlib
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

data_feature = pathlib.Path("../../data/04_feature")
data_output = pathlib.Path("../../data/07_model_output")
fig_output = pathlib.Path("../../docs/figures")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Please input round 1 and round 2 HS results.")
    parser.add_argument('--r1', type=str, required=True,
                        help="Please enter the path for HS round 1 csv.")
    parser.add_argument('--r2', type=str, required=True,
                         help="Please enter the path for HS round 2 csv.")
    parser.print_help()
    return parser.parse_args()

def result_cleaning(r1_path:pathlib.Path, r2_path:pathlib.Path) -> pd.DataFrame:
    with open(r1_path) as round_1:
        r1_df = pd.read_csv(round_1, sep = ",")
        r1_df.columns = [f"{col}_r1" if col not in ["Issue", "Issue_cat"] else col for col in r1_df.columns]

    with open(r2_path) as round_2:
        r2_df = pd.read_csv(round_2, sep = ",")
        r2_df.columns = [f"{col}_r2" if col not in ["Issue", "Issue_cat"] else col for col in r2_df.columns]

    HS_combine = pd.merge(r1_df, r2_df, on=["Issue", "Issue_cat"], how="inner", suffixes=["_r1", "_r2"])
    rank_cols = HS_combine.filter(regex="(?i)^rank", axis = 1).columns
    HS_combine.drop(rank_cols, axis = 1, inplace = True)

    return HS_combine


def main():
    # Read path into python file
    args = parse_arguments()
    r1_path = args.r1
    r2_path = args.r2

    HS_combine = result_cleaning(r1_path, r2_path)
    print(HS_combine.sample(n=8))



if __name__ == "__main__":
    main()
