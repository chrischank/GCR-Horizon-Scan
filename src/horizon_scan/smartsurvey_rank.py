#####################################
#SmartSurvey horizon scan r1 ranking#
#Maintainer: Christopher Chan       #
#Date: 2024-02-14                   #
#Version: 0.0.1                     #
#####################################

import os, sys, re
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Input parameters")
    parser.add_argument("--r1", type=Path, required=True,
                        help='Please provide the round one csv file.')

    parser.print_help()
    return parser.parse_args()

def rank_median(df_path:Path) -> pd.DataFrame:
    with open(df_path) as r_1:
        r1_df = pd.read_csv(r_1, sep=",")
        r1_score = r1_df.select_dtypes(include=[np.number])
        r1_score = pd.merge(r1_df.iloc[:, 0:2], r1_score, how="inner")

        assert r1_score.shape[1] == 96+2

        issue_ls = r1_score.columns.tolist()
        issue_ls2 = [col for col in issue_ls if col not in ["UserID", "Name"]]
        
        print(issue_ls2)

        r1_long = pd.melt(r1_score, id_vars=["Name"], value_vars=issue_ls2,
                          var_name="Issue", value_name="Score")
        r1_rank = r1_long.groupby("Issue")["Score"].rank(ascending=False, method='average')

        print(r1_long.head())
        r1_long.to_csv("../../data/02_intermediate/tidy_r1_score.csv", sep=",")
        r1_rank.to_csv("../../data/02_intermediate/r1_rank.csv", sep=",")

    return r1_long

#def plotting():


def main():
    # Load argparse
    args = parse_arguments()
    r1_path = args.r1
    rank_median(r1_path)

if __name__ == "__main__":
    main()
