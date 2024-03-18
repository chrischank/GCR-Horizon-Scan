#####################################
#SmartSurvey horizon scan r1 ranking#
#Maintainer: Christopher Chan       #
#Date: 2024-02-18                   #
#Version: 0.2.3                     #
#####################################

import os, sys, re
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import date, datetime

# data paths
data_raw = Path("./data/01_raw")
data_intermediate = Path("./data/02_intermediate")
data_output = Path("./data/07_model_output")
figure_output = Path("./docs/figures/round_1")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Input parameters")
    parser.add_argument("--r1", type=Path, required=True,
                        help='Please provide the round one csv file.')

    parser.print_help()
    return parser.parse_args()

def rank_median(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r1_score = df.select_dtypes(include=[np.number])
    r1_score = pd.merge(df.iloc[:, 0:2], r1_score, how="inner")

    assert r1_score.shape[1] == 96+2

    issue_ls = r1_score.columns.tolist()
    issue_ls2 = [col for col in issue_ls if col not in ["UserID", "Name"]]
    print(issue_ls2)

    r1_long = pd.melt(r1_score, id_vars=["Name"], value_vars=issue_ls2,
                      var_name="Issue", value_name="Score")
    r1_long["Rank"] = r1_long.groupby("Name")["Score"].rank(ascending=False, method='average')
    r1_long.sort_values(by=['Name', 'Score'], inplace=True, ascending=False)

    print(r1_long.head())
    r1_long.to_csv(f"{data_intermediate}/{datetime.now().isoformat()}_tidy_r1_score.csv", sep=",")

    # Take the median of each issue ranks
    median_rankDF = r1_long.groupby("Issue")["Rank"].median()
    median_rankDF = median_rankDF.to_frame(name="Median_rank")
    median_rankDF["Final_rank"] = median_rankDF["Median_rank"].rank(ascending=True, method='average')
    median_rankDF.sort_values(by="Final_rank", inplace=True)

    print(median_rankDF.head())

    return r1_long, median_rankDF

def neglected_heardof(df:pd.DataFrame, r1_long:pd.DataFrame) -> pd.DataFrame:
    r1_neg_heard = df[[col for col in df.columns if not col.startswith('Unnamed')]]
    col_drop = [0, 1, 2]
    r1_neg_heard.drop(r1_neg_heard.columns[col_drop], axis=1, inplace=True)
    #r1_neg_heard = pd.concat([df.iloc[:, 0:2], r1_neg_heard], axis=1)
    print(r1_neg_heard.head())

    # Create new pdf with Issues and % calculation
    r1_neg_heard2 = pd.DataFrame({
        "Issue": r1_long["Issue"].unique(),
        "Neglected_perc_YES": None,
        "Heard_of_perc_YES": None})

    neg_ls = []
    heardof_ls = []

    # Fill in the percentage calculation
    for i in r1_neg_heard2["Issue"]:
        neg_idx = r1_neg_heard.columns.get_loc(i) + 1
        heardof_idx = r1_neg_heard.columns.get_loc(i) + 2

        neg_count = (r1_neg_heard.iloc[:, neg_idx] == "Yes").sum(axis=0)
        heardof_count = (r1_neg_heard.iloc[:, heardof_idx] == "Yes").sum(axis=0)

        neg_ls.append(neg_count/len(r1_long["Name"].unique()) * 100)
        heardof_ls.append(heardof_count/len(r1_long["Name"].unique()) * 100)

    r1_neg_heard2["Neglected_perc_YES"] = neg_ls
    r1_neg_heard2["Heard_of_perc_YES"] = heardof_ls


    return r1_neg_heard2


def donut_plot(df:pd.DataFrame, r1_long:pd.DataFrame):
    df["neg_count"] = df["Neglected_perc_YES"] / 100 * len(r1_long["Name"].unique())
    df["heardof_count"] = df["Heard_of_perc_YES"] / 100 * len(r1_long["Name"].unique())

    for idx, issue in df.iterrows():
        i_name = issue["Issue"]
        i_name2 = i_name.replace(" ", "_").replace("/", "or")

        fig, ax = plt.subplots(1,2, figsize = (25, 10))

        donut_negC = ["r", "g"]
        donut_hfC = ["b", "y"]

        plt.rcParams["text.color"] = "darkblue"
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams.update({'font.size': 12})

        # Plotting the first donut plot
        ax[0].pie([issue["neg_count"], len(r1_long["Name"].unique()) - issue["neg_count"]],
                  labels=["Yes", "No"], colors=donut_negC, autopct="%0.0f%%",
                  wedgeprops={"linewidth": 7, "edgecolor": "white"})
        donut_neg = plt.Circle((0, 0), 0.7, color="white")
        ax[0].add_artist(donut_neg)
        ax[0].set_title("Neglected")

        # Plotting the second donut plot
        ax[1].pie([issue["heardof_count"], len(r1_long["Name"].unique()) - issue["heardof_count"]],
                  labels=["Yes", "No"], colors=donut_hfC, autopct="%0.0f%%",
                  wedgeprops={"linewidth": 7, "edgecolor": "white"})
        donut_hf = plt.Circle((0, 0), 0.7, color="white")
        ax[1].add_artist(donut_hf)
        ax[1].set_title("Heard of")

        plt.suptitle(f"{i_name}")

        plt.tight_layout()

        plt.savefig(f"{figure_output}/{i_name2}_neg_HF.png")


def main():
    # Load argparse
    args = parse_arguments()
    r1_path = args.r1

    with open(r1_path) as r_1:
        r1_df = pd.read_csv(r_1, sep=',')
        r1_long, median_rankDF = rank_median(r1_df)
        r1_neg_heard = neglected_heardof(r1_df, r1_long)

        median_rankDF.to_csv(f"{data_intermediate}/{datetime.now().isoformat()}_r1_medianrank.csv", sep=",")
        r1_neg_heard.to_csv(f"{data_intermediate}/{datetime.now().isoformat()}_r1_neg_heardof.csv", sep=",")

        r1_tally = pd.merge(median_rankDF, r1_neg_heard, how="inner", on="Issue")
        r1_tally.to_csv(f"{data_output}/{datetime.now().isoformat()}_r1_tally.csv", sep=",")

        donut_plot(r1_neg_heard, r1_long)

if __name__ == "__main__":
    main()
