
# this program is used to analyze data from the helium network
# it imports a pickle dump from transaction_reader

import pickle
import code
from transaction_reader import HeliumNode, HeliumTransaction, WitnessTransaction

def get_locations(node_dict):
    location_list = []
    for node in node_dict:
        location_list.append(node_dict[node].location)
    return location_list
    # for xcx node_list, there are 474228 unique locations among 476020 nodes

# def get_distance_from_witnesses(node):
#     location = node.location
#     for transaction in node.witness_transactions:
#         for witness in node.witness_transactions[transaction].witness_dict:


if __name__ == "__main__":
    file_path = 'data_xcx.csv'  # Replace with the path to your CSV file

    # store the node_dict so we don't have to re-run and wait each time!
    pickle_file = 'node_dict_xcx.pickle'
    with open(pickle_file, 'rb') as file:
        node_dict = pickle.load(file)


    for node in node_dict:
        node_dict[node].get_witness_distances()

    location_list = get_locations(node_dict)
    print(len(list(set(location_list)))) # get a unique list for an accurate count of unique locations
    print(len(node_dict))

    # Open an interactive Python terminal
    code.interact(local=locals())