from FairRank import *

""" Running the Cherry Blossom Experiments """

""" DOWNSAMPLING CHERRY BLOSSOM """
'''
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split(".")[0]

dataset_path = "./FairRank/Datasets/" + "cherry_blossom" + "/Cleaned/Cleaned_" + experiment_name + ".csv"
print(dataset_path)
GT = pd.read_csv(dataset_path)
GT_male = pd.DataFrame(columns=GT.columns)
GT_female = pd.DataFrame(columns=GT.columns)
num_males = 0
num_females = 0

for index, row in GT.iterrows():
    if (GT.iloc[index, 2]) == 0:
        GT_male = GT_male.append(row)
        num_males = num_males + 1
    elif (GT.iloc[index, 2]) == 1:
        GT_female = GT_female.append(row)
        num_females = num_females + 1

GT_male.sort_values(by='pace_sec')
print(GT_male)
GT_female.sort_values(by='pace_sec')
print(GT_female)

print("Number of Males: ", num_males)
p_males = num_males/19961
print("Proportion of Males: ", p_males)
# new_num_males = p_males*12500
# print("Medium Downsampled Males: ", new_num_males)
print("Small Downsampled Males: ", p_males*5000)

print("Number of Females: ", num_females)
p_females = num_females/19961
print("Proportion of Females: ", p_females)
# new_num_females = p_females*12500
# print("Medium Downsampled Females: ", new_num_females)
print("Small Downsampled Females: ", p_females*5000)

GT_male.to_csv("./FairRank/Datasets/cherry_blossom/male_cherry_blossom_large.csv")
GT_female.to_csv("./FairRank/Datasets/cherry_blossom/female_cherry_blossom_large.csv")
'''

# clean()

# CalculateInitialMetrics()
# split()
# train()
# BehindTheName()
# NameSor()
# GenderAPI()
# rank_ground_truth()
# rank_inferred()
# DetConstSort()
# CalculateResultsMetrics()

skew1 = "./FairRank/Datasets/cherry_blossom/Ranked/GroundTruth_Ranked/GroundTruth_Ranked(num_iterations=5000,gamma=1.0)_cherry_blossom.csv"
skew2 = "./FairRank/Datasets/cherry_blossom/Ranked/DetConstSort_RankedDetConstSort_Ranked(num_iterations=5000,gamma=0.0)_cherry_blossom.csv"
group = '1'
avg_pos_dif(skew1, skew2, group)
