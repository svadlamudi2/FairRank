import json
import os
import pickle
import time

import numpy as np
from fairsearchdeltr import Deltr
import pandas as pd
with open('./FairRank/settings.json', 'r') as f:
    settings = json.load(f)


def Train():
    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
    read_file = './FairRank/Datasets/' + filename +'/Training/' + 'Training_' + filename + '.csv'

    if not os.path.isfile(read_file):
        print("This file: " + read_file + "does not exist, check read file options in settings.json")
        return

    START = time.time()

    train_data = pd.read_csv(read_file)
    # Only get the numeric colums in the training dataset and make a new dataframe
    numeric_cols = list(train_data.select_dtypes(include=[np.number]).columns.values)
    formatted_data = train_data[numeric_cols]

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

    # set up the DELTR object, # column name of the protected attribute (index after query and document id)
    if settings["READ_FILE_SETTINGS"]["GENDER_EXPERIMENT"].lower() == "true":
        protected_attribute = "Gender"
    elif settings["READ_FILE_SETTINGS"]["RACE_EXPERIMENT"].lower() == "true":
        protected_attribute = "Race"
    gamma = settings["DELTR_OPTIONS"]["gamma"]  # value of the gamma parameter
    number_of_iterations = settings["DELTR_OPTIONS"]["num_iterations"]  # number of iterations the training should run
    standardize = True if settings["DELTR_OPTIONS"]["standardize"].lower() == "true" else False  # let's apply standardization to the features

    # create the Deltr object
    dtr = Deltr(protected_attribute, gamma, number_of_iterations, standardize=standardize)

    print(formatted_data)


    # train the model
    print("Beginning to train the model with parameters: \n"
          "Protected Attribute: " + protected_attribute + "\n"
          "Gamma: " + str(gamma) + "\n"
          "Number of Iterations: " + str(number_of_iterations) + "\n"
          "Standardize: " + str(standardize))
    print("This could take a while...")


    dtr.train(formatted_data)

    print("SUCCESS! Time Taken: ", time.time() - START)

    FILE_PATH = "./FairRank/Models/" + "(num_iterations=" + str(number_of_iterations) + ",gamma=" + str(gamma) + ")" + filename + ".obj"
    # make file in 'Test Datasets' folder, pickle the model, and dump it there
    file = open(FILE_PATH, "wb")
    pickle.dump(dtr, file)
    print("SAVED MODEL TO PATH: " + FILE_PATH)


# train()