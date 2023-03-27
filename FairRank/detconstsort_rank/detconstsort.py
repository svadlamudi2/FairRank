import json
import math
import os
import re
from collections import defaultdict as ddict
import operator
import pandas as pd
import FairRank.data_ranking.rank as rank
with open('./FairRank/settings.json', 'r') as f:
    settings = json.load(f)


def detconstsort(a, k_max, p):
    scores = []
    for a_i in a.keys():
        for i_d, score in a[a_i].items():
            scores.append((a_i, i_d, score))
    attributes = a.keys()
    attribute_scores = {}

    # create and initialize counter for each attribute value
    counts_ai = {}
    minCounts_ai = {}
    totalCounts_ai = {}
    for a_i in a.keys():
        counts_ai[a_i] = 0
        minCounts_ai[a_i] = 0
        totalCounts_ai[a_i] = len(a[a_i])

    re_ranked_attr_list = {}
    re_ranked_score_list = {}
    maxIndices = {}

    lastEmpty = 0
    k = 0

    for i, a_i in enumerate(attributes):
        counts_ai[a_i] = 0
        minCounts_ai[a_i] = 0
        totalCounts_ai[a_i] = sum([1 for s in scores if s[0] == a_i])
        attribute_scores[a_i] = [(s[2], s[1]) for s in scores if
                                 s[0] == a_i]

    # print(attribute_scores)

    while lastEmpty <= k_max:

        if lastEmpty == len(scores):
            break

        k += 1
        tempMinAttrCount = ddict(int)
        changedMins = {}
        for a_i in attributes:
            tempMinAttrCount[a_i] = math.floor(k * p[a_i])
            if minCounts_ai[a_i] < tempMinAttrCount[a_i] and minCounts_ai[a_i] < totalCounts_ai[a_i]:
                changedMins[a_i] = attribute_scores[a_i][counts_ai[a_i]]

        if len(changedMins) != 0:
            ordChangedMins = sorted(changedMins.items(), key=lambda x: x[1][0], reverse=True)
            for a_i in ordChangedMins:
                re_ranked_attr_list[lastEmpty] = a_i[0]
                lastEmpty = int(lastEmpty)
                # print('here', attribute_scores[a_i[0]][counts_ai[a_i[0]]])
                re_ranked_score_list[lastEmpty] = attribute_scores[a_i[0]][counts_ai[a_i[0]]]
                maxIndices[lastEmpty] = k
                start = lastEmpty
                while start > 0 and maxIndices[start - 1] >= start and re_ranked_score_list[start - 1][0] < \
                        re_ranked_score_list[start][0]:
                    swap(re_ranked_score_list, start - 1, start)
                    swap(maxIndices, start - 1, start)
                    swap(re_ranked_attr_list, start - 1, start)
                    start -= 1
                counts_ai[a_i[0]] += 1
                lastEmpty += 1
            minCounts_ai = dict(tempMinAttrCount)

    re_ranked_attr_list = [re_ranked_attr_list[i] for i in sorted(re_ranked_attr_list)]
    re_ranked_score_list = [re_ranked_score_list[i] for i in sorted(re_ranked_score_list)]

    return re_ranked_attr_list, re_ranked_score_list


def swap(temp_list, pos_i, pos_j):
    temp = temp_list[pos_i]
    temp_list[pos_i] = temp_list[pos_j]
    temp_list[pos_j] = temp


def wrapper(url):
    """
    This is the wrapper code to convert detlr output to input 1 for detconstsort_rank
    :param url: url pointing to deltr output
    :return:
    """
    a = {}
    df = pd.read_csv(url)
    dff = df.groupby('Gender')
    for row in df['Gender']:
        a[row] = dict(zip(dff.get_group(row).doc_id, dff.get_group(row).judgement))
    return a


def getdist(df):
    # Given the ranked dataframe, return the true protected attr dist as a dictionary
    d = {}
    for index, row in df.iterrows():
        if row["Gender"] not in d:
            d[row['Gender']] = 1
        else:
            d[row['Gender']] += 1
    for attr in d:
        d[attr] = d[attr] / len(df)
    return d


def find_unaware_ranked(file):
    match = re.search('gamma=0.0', file)
    if match:
        return True
    else:
        return False


def infer_with_detconstsort(file, inferred = False):
    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]

    write_path = './FairRank/Datasets/' + filename + '/Ranked/DetConstSort_Ranked'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    ranked_dict = {}
    if inferred:
        params = file.split("Inferred_Ranked(")[1]
        write_file = write_path + '/DetConstSort_Ranked(' + params
    else:
        params = re.findall(r"\(([^)]+)\)", file)[0]
        write_file = write_path + '/DetConstSort_Ranked(' + params + ')_' + \
                     os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0] + '.csv'

    data = pd.read_csv(file)
    a = wrapper(file)
    p = getdist(data)
    k_max = len(data.index)

    result = detconstsort(a, k_max, p)
    result_genders = result[0]
    result_scores = result[1]

    for i in range(k_max):
        if result_scores[i][1] not in ranked_dict.keys():
            ranked_dict[result_scores[i][1]] = [result_scores[i][1], result_genders[i], result_scores[i][0]]
        else:
            print("There are duplicates in the ranking, something went wrong.")
            return

    rank.writeRanked(write_file, ranked_dict)


def DetConstSort():
    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
    gt_ranked = filter(find_unaware_ranked, rank.get_files('./FairRank/Datasets/' + filename + '/Ranked/GroundTruth_Ranked'))
    inferred_ranked = filter(find_unaware_ranked, rank.get_files('./FairRank/Datasets/' + filename + '/Ranked/Inferred_Ranked'))

    for file in gt_ranked:
        infer_with_detconstsort(file)
    for file in inferred_ranked:
        print(file)
        infer_with_detconstsort(file, inferred=True)



