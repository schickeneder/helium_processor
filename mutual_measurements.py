import pickle
import numpy as np
import matplotlib.pyplot as plt
from plot_distance_data import plot_distances

# analyze measurements from mutual pairs where two nodes witness one another
# [txlocation, location, distance_from_tx_location [,txpwr] [,rxpwr] [,timestamp] [,txgateway] [,wgateway]]

"""
xcx dataset:

Mutual_nodes:
1215300 1110482
Value: 3, Count: 10908
Value: 1, Count: 1025503
Value: 2, Count: 70425
Value: 4, Count: 2563
Value: 6, Count: 238
Value: 5, Count: 726
Value: 9, Count: 13
Value: 8, Count: 24
Value: 7, Count: 76
Value: 11, Count: 2
Value: 12, Count: 2
Value: 10, Count: 1
Value: 16, Count: 1

Mutual_locs:
1215300 1110143
Value: 3, Count: 10945
Value: 1, Count: 1024951
Value: 2, Count: 70568
Value: 4, Count: 2578
Value: 6, Count: 240
Value: 5, Count: 741
Value: 9, Count: 13
Value: 8, Count: 25
Value: 7, Count: 76
Value: 11, Count: 2
Value: 12, Count: 2
Value: 10, Count: 1
Value: 16, Count: 1
"""

# returns a list of keys representing two nodes in which mutual witness events occurred
def count_mutual_witnesses(distances):
    mutual_nodes = {}
    for row in distances:
        if str(row[-2]) < str(row[-1]):
            new_key = row[-2] + '_' + row[-1]
        else:
            new_key = row[-1] + '_' + row[-2]
        if new_key not in mutual_nodes:
            mutual_nodes[new_key] = 1
        else:
            mutual_nodes[new_key] += 1

    return mutual_nodes

# returns a list of keys representing two *locations* in which mutual witness events occurred
def count_mutual_locations(distances):
    mutual_locs = {}
    for row in distances:
        if str(row[0]) < str(row[1]):
            new_key = row[0] + '_' + row[1]
        else:
            new_key = row[1] + '_' + row[0]
        if new_key not in mutual_locs:
            mutual_locs[new_key] = 1
        else:
            mutual_locs[new_key] += 1

    return mutual_locs

# returns a list of keys representing locations in which nodes exist
# TODO not working yet..
def count_number_locations(distances):
    locs = {}
    for row in distances:
        if row[0] not in mutual_locs:
            mutual_locs[row[0]] = 1
        else:
            mutual_locs[row[0]] += 1

    return locs

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

# returns a list of node pairs in which n or more mutual witness events occurred
# a pair (or mutual) looks like: 112oz...F6PHJ4772TPYXip_11qqKtKcbYD..tuzuPwc6wBmfvu or <gateway1>_<gateway2> where
# '<gateway1>' < '<gateway2>' in a string comparison
def return_pairs_n_counts(mutuals, n):
    pairs = []
    for pair, counts in mutuals.items():
        if counts >= n:
            pairs.append(pair)

    return pairs

# returns a distances information associated with a list of pairs
# in the form a dictionary with keys as pairs and values as lists like distances
def get_distance_info_from_pairs(distances, pairs):
    pairs_distance_info = {}
    for pair in pairs:
        pair_info = []
        pair1, pair2 = pair.split('_')
        for row in distances:
            if pair1 in row and pair2 in row:
                #print("both {} and {} in row {}!".format(pair1,pair2,row))
                pair_info.append(row)
        pairs_distance_info[pair] = pair_info

    return pairs_distance_info

if __name__ == "__main__":


    pickle_file = 'measurements_xcx.pickle'
    with open(pickle_file, 'rb') as file:
        distances = pickle.load(file)

    # for row in distances:
    #     print(row, (row[-1] > row[-2]))

    # mutuals = count_mutual_witnesses(distances)
    # print(len(distances),len(mutuals))
    mutual_locs = count_mutual_locations(distances)
    print(len(distances),len(mutual_locs))

    # for row in mutuals:
    #     print(row)

    #mutuals = recompute_mutuals(mutuals,distances)
    value_counts = histogram_mutual_witness_events(mutual_locs)
    # Print the counts
    for value, count in value_counts.items():
        print(f"Value: {value}, Count: {count}")


    # pairs = return_pairs_n_counts(mutuals,5)
    #
    # pairs_distance_info = get_distance_info_from_pairs(distances,pairs)
    #
    # # store the node_dict so we don't have to re-run and wait each time!
    # pickle_file = 'pairs_distances.pickle'
    # with open(pickle_file, 'wb') as file:
    #     pickle.dump(pairs_distance_info,file)
    #
    #
    # #print(pairs_distance_info[pairs[0]])
    # for pair in pairs:
    #     print(pair)
    #     plot_distances(pairs_distance_info[pair])

    # plt.hist(values, bins=20, edgecolor='black')  # You can adjust the number of bins as needed
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Values in the Dictionary')
    # plt.show()