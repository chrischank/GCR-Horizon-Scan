############################
#OI HS Analysis Script     #
#Maintainer: Timur Burkanov#
#Date: 2024-04-17          #
#Version: 0.0.1            #
############################

import os
import matplotlib
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import FancyArrowPatch

data_feature = Path("../../data/04_feature")
data_output = Path("../../data/07_model_output")
fig_output = Path("../../docs/figures")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Please input round 1 and round 2 HS results.")
    parser.add_argument('--r1', type=str, required=True,
                        help="Please enter the path for HS round 1 csv.")
    parser.add_argument('--r2', type=str, required=True,
                         help="Please enter the path for HS round 2 csv.")
    parser.print_help()
    return parser.parse_args()

def result_cleaning(r1_path:Path, r2_path:Path) -> pd.DataFrame:
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

def score_sample(df:pd.DataFrame) -> pd.DataFrame:
    misc_cols = df.filter(regex="rank|score").columns
    HS_justScore = df.drop(misc_cols, axis = 1)

    return HS_justScore.sample(n = 8)

def score_Ldf(df:pd.DataFrame) -> pd.DataFrame:
    justScore_Ldf = pd.melt(df, id_vars=["Issue", "Issue_cat"], var_name='Participant', value_name='Score')
    justScore_Ldf["HS_round"] = justScore_Ldf["Participant"].str.extract('(_r\d)').replace({'_r1': 1, '_r2': 2})
    justScore_Ldf["Participant"] = justScore_Ldf["Participant"].str.replace('(_r\d)', '', regex=True)
    print(justScore_Ldf.loc[:, ['Score', 'HS_round']].groupby('HS_round').describe())

    return justScore_Ldf

def plot_model(df:pd.DataFrame) -> (list, list):
    sns.set_style('darkgrid', {'axes.facecolor': '.85'})

    fig, ax = plt.subplots(6, 2, figsize=(15, 30))

    axes = ax.flatten()

    r1_exp = []
    r2_exp = []

    print(df['Issue_cat'].unique())


    for idx, cat in enumerate(df['Issue_cat'].unique()):
        palette_ls = ['Set3', 'Set2', 'Paired', 'Accent', 'Dark2', 'Set1',
                    'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']
        linestyle_ls = ['solid', 'dotted', 'dashed', 'dashdot', 'solid',
                        'dotted', 'dashed', 'dashdot', 'solid', 'dotted',
                        'dashed', 'dashdot']
        colour_ls = ['black', 'red', 'gold', 'limegreen', 'cyan', 'indigo',
                    'lightpink', 'black', 'red', 'gold', 'limegreen', 'cyan']

        justScore_cat = df[df['Issue_cat'] == cat]

        # Inverse WLS that uses variance as weights
        y = justScore_cat['Score']
        x = justScore_cat['HS_round']
        X_sm = sm.add_constant(x)
        variance = justScore_cat.groupby('HS_round')['Score'].var()
        in_wt = justScore_cat['HS_round'].map(variance)

        # Create model, fit, and print results
        inv_WLS = sm.WLS(y, X_sm, weights = in_wt)
        res_inv = inv_WLS.fit()
        inv_ypred = res_inv.predict()
        print("\nModified WLS results")
        print(res_inv.summary())

        # Fill DataFrame
        r1_exp.append(inv_ypred[0])
        r2_exp.append(inv_ypred[-1])
        
        splot = sns.scatterplot(justScore_cat, x='HS_round', y='Score',
                                hue='Issue', ax=axes[idx], palette=palette_ls[idx])
        WLSplot = sns.lineplot(justScore_cat, x='HS_round', y=inv_ypred,
                            ax=axes[idx], linestyle=linestyle_ls[idx], color=colour_ls[idx],
                            label=f"modified_WLS: y = {res_inv.params[1]:.2f}X + {res_inv.params[0]:.2f}'")

        axes[idx].set_title(f"Issue Category: {cat}")
        axes[idx].set_xticks([1, 2])
        axes[idx].set_ylim(1-1, 1000)

        if idx % 2 == 0:
            # Left side subplots
            axes[idx].legend(bbox_to_anchor = (-0.1, 1), ncol=1)
        else:
            # Right side subplots
            axes[idx].legend(bbox_to_anchor=(1, 1), ncol=1)

    plt.tight_layout()
    plt.savefig(f"{fig_output}/WLSissueCat_subplots.png")
    plt.show()

    return r1_exp, r2_exp


def WLS_expected_score(df:pd.DataFrame, r1_exp:list, r2_exp:list) -> pd.DataFrame:
    # DataFrame to collect inv_WLS ypred expected values
    WLS_expDF = pd.DataFrame({
        "Issue_cat": df["Issue_cat"].unique(),
        "round_1": r1_exp,
        "round_2": r2_exp
    })

    WLS_expDF_L = pd.melt(WLS_expDF, id_vars="Issue_cat", value_vars=["round_1", "round_2"], var_name="HS_round", value_name="Expected_Score")

    return WLS_expDF_L

def lollipop(df:pd.DataFrame):
    # Lollipop chart
    sns.set_style('whitegrid', {'axes.facecolor': '0.98'})

    fig, ax = plt.subplots(figsize = (10, 7))

    #lolli = sns.scatterplot(WLS_expDF_L, x="Expected_Score", y="Issue_cat",
    #                        hue="HS_round", palette="Set2")

    for idx, cat in enumerate(df["Issue_cat"].unique()):
        temp_df = df[df["Issue_cat"] == cat]

        r1_score = temp_df[temp_df["HS_round"] == "round_1"]["Expected_Score"].values[0]
        r2_score = temp_df[temp_df["HS_round"] == "round_2"]["Expected_Score"].values[0]

        if r2_score > r1_score:
            cane_col = "#91bfdb"
            plt.gca().add_patch(FancyArrowPatch((r1_score, idx), (r2_score, idx),
                                        arrowstyle='->', color='#91bfdb', mutation_scale=15))
        else:
            cane_col = "#fc8d59"        
            plt.gca().add_patch(FancyArrowPatch((r2_score, idx), (r1_score, idx),
                                        arrowstyle='<-', color='#fc8d59', mutation_scale=15))


        sns.lineplot(temp_df, x="Expected_Score", y="Issue_cat",
                    color=cane_col, zorder=0)

    lolli = sns.scatterplot(df, x="Expected_Score", y="Issue_cat",
                            color="#000000", palette="Set2", zorder=1,
                            style="HS_round", s=40)


    lolli.yaxis.grid(False)
    handles, _ = lolli.get_legend_handles_labels()
    new_handles = ['Round 1', 'Round 2']
    lolli.legend(handles, new_handles, loc="lower right")
    #lolli.set_xlim(1, 1000)
    plt.xticks(list(range(300, 900, 100)))

    sns.despine(left=True, bottom=True)
    plt.ylabel("Issue Category")
    plt.xlabel("Predicted Score")
    plt.title("Weighted Least Square Score shift after the Horizon Scan workshop")

    plt.tight_layout()
    plt.savefig(f"{fig_output}/Deliberation_lolli.png")
    plt.show()

def main():
    # Read path into python file
    args = parse_arguments()
    r1_path = args.r1
    r2_path = args.r2

    HS_combine = result_cleaning(r1_path, r2_path)
    print(HS_combine.sample(n=8))
    HS_justScore = score_sample(HS_combine)
    justScore_Ldf = score_Ldf(HS_justScore)

    r1_exp, r2_exp=plot_model(justScore_Ldf)

    WLS_expDF_L = WLS_expected_score(justScore_Ldf, r1_exp, r2_exp)

    lollipop(WLS_expDF_L)


if __name__ == "__main__":
    main()
