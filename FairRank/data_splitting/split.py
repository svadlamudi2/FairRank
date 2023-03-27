import csv
import random
import json
import os
with open('./FairRank/settings.json', 'r') as f:
    settings = json.load(f)


# Split the original dataset
# readFile: The original dataset
# train_file: The file that the training data needs to be written to
# test_file: The file that the testing data needs to be written to
# train_split: The percent of the original dataset that needs to be in training data (Range: 0-1)
def Split():
    train_split = settings["DATA_SPLIT"]["TRAIN_PCT"]
    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]

    write_path = './FairRank/Datasets/' + filename + '/Training'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    write_path = './FairRank/Datasets/' + filename + '/Testing'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    read_file = './FairRank/Datasets/' + filename + '/Cleaned' + '/Cleaned_' + filename + '.csv'
    train_file = './FairRank/Datasets/' + filename + '/Training' + '/Training_' + filename + '.csv'
    test_file = './FairRank/Datasets/' + filename + '/Testing' + "/Testing_" + filename + '.csv'

    if os.path.isfile(train_file):
        print("This File Has Already Been Split Into Training File at Path: " + train_file)
        return
    if os.path.isfile(test_file):
        print("This File Has Already Been Split Into Testing File at Path: " + test_file)
        return

    # Open the file
    with open(read_file, mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)
        # Skip the headers row
        headers = next(csvFile)
        # Write headers to both train and test data sets
        writeData(train_file, headers)
        writeData(test_file, headers)
        for lines in csvFile:
            if random.random() < train_split:
                writeData(train_file, lines)
            else:
                writeData(test_file, lines)

    print("Success!: Saved Train File to: " + train_file)
    print("Success!: Saved Test File to: " + test_file)

def writeData(write_file, fields):
    # writing to csv file
    with open(write_file, 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)


# split()