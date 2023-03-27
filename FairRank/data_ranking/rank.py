import csv
import re
import json
import pickle
import numpy as np
import pandas as pd
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


models = get_files('./FairRank/Models')
gt = get_files('./FairRank/Datasets/' + experiment_name + '/Testing')
inferred = get_files('./FairRank/Datasets/' + experiment_name + '/Inferred')


def writeRanked(writefile, dict):
    # field names
    fields = ['doc_id', 'Gender', 'judgement']

    # writing to csv file
    with open(writefile, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        for player in dict.keys():
            csvwriter.writerow(dict.get(player))
    print("SUCCESS! Saved to: " + writefile)


def RankGroundTruth():
    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]

    write_path = './FairRank/Datasets/' + filename + '/Ranked/GroundTruth_Ranked'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    for file in gt:
        for model in models:
            params = re.findall(r"\(([^)]+)\)", model)[0]
            write_file = write_path + '/GroundTruth_Ranked(' + params + ')_' + \
                         os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0] + '.csv'
            print("Ranking Ground-Truth File: " + file + " With Model: " + model)
            print("Results will be saved to: " + write_file)
            filehandler = open(model, 'rb')
            DELTR = pickle.load(filehandler)

            test_data = pd.read_csv(file, index_col=False)
            numeric_cols = list(test_data.select_dtypes(include=[np.number]).columns.values)
            formatted_data = test_data[numeric_cols]

            score_column = settings["DELTR_OPTIONS"]["SCORE_COLUMN"]
            lower_better = settings["READ_FILE_SETTINGS"]["LOWER_SCORE_BETTER"].lower()
            normalized = settings["DELTR_OPTIONS"]["NORMALIZE_SCORE_COLUMN"].lower()

            if lower_better == "true" and normalized == "true":
                formatted_data['normalized_score'] = formatted_data.apply(
                    lambda row: (formatted_data[score_column].max() - row[score_column]) / (
                            formatted_data[score_column].max() - formatted_data[score_column].min()), axis=1)
                formatted_data = formatted_data.drop(score_column, axis=1)
            elif lower_better == "false" and normalized == "true":
                formatted_data['normalized_score'] = formatted_data.apply(
                    lambda row: (row[score_column] - formatted_data[score_column].min()) / (
                            formatted_data[score_column].max() - formatted_data[score_column].min()), axis=1)
                formatted_data = formatted_data.drop(score_column, axis=1)

            formatted_data = formatted_data.sample(frac=1)
            print(formatted_data)

            result = DELTR.rank(formatted_data, has_judgment=True)
            print("SUCCESS! Saved to: " + write_file)
            result.to_csv(write_file, index=False)


def RankInferred():
    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]

    write_path = './FairRank/Datasets/' + filename + '/Ranked/Inferred_Ranked'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    gt_dict = {}
    for index, row in pd.read_csv(gt[0]).iterrows():
        if int(row["doc_id"]) not in gt_dict.keys():
            gt_dict[int(row["doc_id"])] = int(row["Gender"])
        else:
            print("There are duplicates in the ranking something went wrong")
            exit(1)

    for file in inferred:
        for model in models:
            params = re.findall(r"\(([^)]+)\)", model)[0]
            write_file = write_path + '/Inferred_Ranked(' + params + ')_' + \
                         os.path.basename(file).split('.')[0] + '.csv'
            print("Ranking Inferred File: " + file + " With Model: " + model)
            print("Results will be saved to: " + write_file)

            filehandler = open(model, 'rb')
            DELTR = pickle.load(filehandler)

            test_data = pd.read_csv(file)
            numeric_cols = list(test_data.select_dtypes(include=[np.number]).columns.values)
            test_data.drop(columns=[col for col in test_data if col not in numeric_cols], inplace=True)
            test_data.drop("Gender", axis=1, inplace=True)

            score_column = settings["DELTR_OPTIONS"]["SCORE_COLUMN"]
            lower_better = settings["READ_FILE_SETTINGS"]["LOWER_SCORE_BETTER"].lower()
            normalized = settings["DELTR_OPTIONS"]["NORMALIZE_SCORE_COLUMN"].lower()

            if lower_better == "true" and normalized == "true":
                test_data['normalized_score'] = test_data.apply(
                    lambda row: (test_data[score_column].max() - row[score_column]) / (
                            test_data[score_column].max() - test_data[score_column].min()), axis=1)
                test_data = test_data.drop(score_column, axis=1)
            elif lower_better == "false" and normalized == "true":
                test_data['normalized_score'] = test_data.apply(
                    lambda row: (row[score_column] - test_data[score_column].min()) / (
                            test_data[score_column].max() - test_data[score_column].min()), axis=1)
                test_data = test_data.drop(score_column, axis=1)

            test_data = test_data.sample(frac=1)
            print(test_data)

            result = DELTR.rank(test_data)
            gt_inferred_combined_dict = {}
            for index, row in result.iterrows():
                if int(row["doc_id"]) not in gt_inferred_combined_dict.keys():
                    gt_inferred_combined_dict[row["doc_id"]] = [str(int(row["doc_id"])),
                                                                str(gt_dict[int(row["doc_id"])]), str(row["judgement"])]
                else:
                    print("There are duplicates in the inferred dictionary something went wrong")
                    exit(1)

            writeRanked(write_file, gt_inferred_combined_dict)
