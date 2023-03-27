import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from ..data_analysis import avgExp, NDCG, NDKL, skew
import csv
import os

with open('./FairRank/settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]


def get_files(directory):
    temp = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            match = re.search(experiment_name, file)
            if match:
                # temp.append(directory + '/' + file)
                temp.append(os.path.join(dirpath, file))
    return temp


def CalculateInitialMetrics():
    """ GET VARIABLES FROM SETTINGS """
    dataset_path = "./FairRank/Datasets/" + experiment_name + "/Cleaned/Cleaned_" + experiment_name + ".csv"
    GT = pd.read_csv(dataset_path)

    print(GT)

    GT_ranking_ids = np.array(GT.iloc[:, 1])  # doc IDs
    GT_group_ids = np.array(GT.iloc[:, 2])  # protected attribute column
    score_col_index = GT.columns.get_loc(os.path.basename(settings["DELTR_OPTIONS"]["SCORE_COLUMN"]))
    GT_score = np.array(GT.iloc[:, score_col_index])  # numerical scoring column from settings.json
    protected_attribute = settings["READ_FILE_SETTINGS"]["DEMO_COL"]  # TODO decide if automating this or not

    """ DIRECTORY MANAGEMENT """
    results_path = Path(
        "./FairRank/Results/" + os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0] + "/Initial")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    """ CALCULATE NDKL """
    ndkl = "NDKL: ", NDKL(GT_ranking_ids, GT_group_ids)

    """ CALCULATE AVERAGE EXPOSURE """
    avg_exp = "Average Exposure: ", avgExp.avg_exp(GT_ranking_ids, GT_group_ids)

    metrics_path = results_path / "initial_metrics.csv"
    with open(metrics_path, 'w') as f_metrics:
        print("Writing to metrics csv.")
        writer = csv.writer(f_metrics)
        writer.writerow(ndkl)
        writer.writerow(avg_exp)

    """ CALCULATE SKEW """
    print("Calculating skew...")
    skew_path = results_path / "initial_skews.csv"
    skew_data = []
    for i in range(1, len(GT) + 1):
        skew_0 = skew(GT_ranking_ids, GT_group_ids, 0, i)
        skew_1 = skew(GT_ranking_ids, GT_group_ids, 1, i)
        skew_data.append([i, skew_0, skew_1])

    print("Finished calculating skews.")
    skew_header = ["Position", "Group 0", "Group 1"]
    with open(skew_path, 'w') as f_skew:
        print("Writing to skews csv.")
        writer = csv.writer(f_skew)
        # write the header
        writer.writerow(skew_header)

        # write the data
        writer.writerows(skew_data)

    print("Skews written to csv.")

    """ CALCULATE NDCG """
    print("Calculating NDCG...")
    ndcg_path = results_path / "initial_ndcg.csv"
    ndcg_data = []
    for i in range(1, len(GT) + 1):
        ndcg = NDCG(GT_ranking_ids, GT_score, i)
        ndcg_data.append([i, ndcg])

    print("Finished calculating NDCG.")
    ndcg_header = ["Position", "NDCG"]
    with open(ndcg_path, 'w') as f_ndcg:
        print("Writing to NDCG csv.")
        writer = csv.writer(f_ndcg)
        # write the header
        writer.writerow(ndcg_header)

        # write the data
        writer.writerows(ndcg_data)

    print("NDCG written to csv.")


def CalculateResultsMetrics():
    print(experiment_name)
    ranked = get_files('./FairRank/Datasets/' + experiment_name + '/Ranked')
    for file in ranked:
        print(file)
        calc_metrics_util(file)


def calc_metrics_util(dataset_path):
    print("utility function")

    """ GET VARIABLES """
    ranking = pd.read_csv(dataset_path)
    rank_name = dataset_path.split('/')[6]
    print(rank_name)

    ranking_ids = np.array(ranking.iloc[:, 0])  # doc IDs
    group_ids = np.array(ranking.iloc[:, 1])  # protected attribute column
    score_col_index = 2  # score column index
    score = np.array(ranking.iloc[:, score_col_index])  # numerical scoring column from settings.json

    """ DIRECTORY MANAGEMENT """
    results_path = Path(
        "./FairRank/Results/" + os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0] + "/" + rank_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    """ CALCULATE NDKL """
    ndkl = "NDKL: ", NDKL(ranking_ids, group_ids)

    """ CALCULATE AVERAGE EXPOSURE """
    avg_exp = "Average Exposure: ", avgExp.avg_exp(ranking_ids, group_ids)

    metrics_path = results_path / "metrics.csv"
    with open(metrics_path, 'w') as f_metrics:
        print("Writing to metrics csv.")
        writer = csv.writer(f_metrics)
        writer.writerow(ndkl)
        writer.writerow(avg_exp)

    """ CALCULATE SKEW """
    print("Calculating skew...")
    skew_path = results_path / "skews.csv"
    skew_data = []
    for i in range(1, len(ranking) + 1):
        skew_0 = skew(ranking_ids, group_ids, 0, i)
        skew_1 = skew(ranking_ids, group_ids, 1, i)
        skew_data.append([i, skew_0, skew_1])

    print("Finished calculating skews.")
    skew_header = ["Position", "Group 0", "Group 1"]
    with open(skew_path, 'w') as f_skew:
        print("Writing to skews csv.")
        writer = csv.writer(f_skew)
        # write the header
        writer.writerow(skew_header)

        # write the data
        writer.writerows(skew_data)

    print("Skews written to csv.")

    """ CALCULATE NDCG """
    print("Calculating NDCG...")
    ndcg_path = results_path / "ndcg.csv"
    ndcg_data = []
    for i in range(1, len(ranking) + 1):
        ndcg = NDCG(ranking_ids, score, i)
        ndcg_data.append([i, ndcg])

    print("Finished calculating NDCG.")
    ndcg_header = ["Position", "NDCG"]
    with open(ndcg_path, 'w') as f_ndcg:
        print("Writing to NDCG csv.")
        writer = csv.writer(f_ndcg)
        # write the header
        writer.writerow(ndcg_header)

        # write the data
        writer.writerows(ndcg_data)

    print("NDCG written to csv.")
