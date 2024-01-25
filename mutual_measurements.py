import pickle
import numpy as np
import matplotlib.pyplot as plt

# analyze measurements from mutual pairs where two nodes witness one another
# [txlocation, location, distance_from_tx_location [,txpwr] [,rxpwr] [,timestamp] [,txgateway] [,wgateway]]

"""
xcx dataset:
witness events: 1215300, witness events between unique set of nodes: 1143784
breakout of unique mutual witness counts
Value: 3, Count: 6536
Value: 1, Count: 1082249
Value: 2, Count: 53496
Value: 4, Count: 1161
Value: 5, Count: 266
Value: 6, Count: 60
Value: 7, Count: 12
Value: 8, Count: 3
Value: 9, Count: 1
"""

def count_mutual_witnesses(distances):
    mutuals = {}
    for row in distances:
        if row[-2] < row[1]:
            new_key = row[-2] + '_' + row[-1]
        else:
            new_key = row[-1] + '_' + row[-2]
        if new_key not in mutuals:
            mutuals[new_key] = 1
        else:
            mutuals[new_key] += 1

    return mutuals

def histogram_mutual_witness_events(mutuals):
    values = list(mutuals.values())

    # Count occurrences of each value
    value_counts = {}
    for value in values:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

    return value_counts

# returns a list of node pairs in which n mutual witness events occurred
# a pair looks like: 112oz...F6PHJ4772TPYXip_11qqKtKcbYD..tuzuPwc6wBmfvu or <gateway1>_<gateway2> where
# '<gateway1>' < '<gateway2>' in a string comparison
def return_pairs_n_counts(mutuals, n):
    pairs = []
    for mutual, counts in mutuals.items():
        if counts == n:
            pairs.append(mutual)

    return pairs

# returns a distances information associated with a list of pairs
def get_distance_info_from_pairs(distances, pairs):
    pairs_distance_info = {}
    for pair in pairs:
        pair_info = []
        pair1, pair2 = pair.split('_')
        for row in distances:
            if pair1 in row and pair2 in row:
                print("both {} and {} in row {}!".format(pair1,pair2,row))


if __name__ == "__main__":


    pickle_file = 'measurements_xcx.pickle'
    with open(pickle_file, 'rb') as file:
        distances = pickle.load(file)

    # for row in distances:
    #     print(row, (row[-1] > row[-2]))

    mutuals = count_mutual_witnesses(distances)
    print(len(distances),len(mutuals))
    # for row in mutuals:
    #     print(row)

    value_counts = histogram_mutual_witness_events(mutuals)
    # Print the counts
    for value, count in value_counts.items():
        print(f"Value: {value}, Count: {count}")

    pairs = return_pairs_n_counts(mutuals,7)
    for pair in pairs:
        print(pair)

    get_distance_info_from_pairs(distances,pairs)

    # plt.hist(values, bins=20, edgecolor='black')  # You can adjust the number of bins as needed
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Values in the Dictionary')
    # plt.show()