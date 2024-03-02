import os
import gzip
import csv
import sys

from transaction_reader import load_witnesses
import code
import pickle
import random

# Read in raw data or pickle files, process and save for quick use in other applications
# pickle filename key
# a distance row is structured like:
# [txlocation, location, distance_from_tx_location [,txpwr] [,rxpwr] [,timestamp] [,txgateway] [,wgateway]]

transaction_filepath = r'D:\blockchain-etl-export\transactions'
# try to process all .csv files in this path, even nested



# (Chatgpt3.5 generated)
# returns a list of all .csv files in path, including recursive folders
# list contains full path like 'D:\blockchain-etl-export\transactions\unzipped\data_xaf.csv\data_xaf.csv'
def find_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv.gz'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def open_csv_gz_load_witnesses(filepath,node_dict):
    filepath_parts = filepath.split('.')
    csv.field_size_limit(2**30) # threw error because exceeded size and sys.maxsize is 2*64-1, also way too big
    if filepath_parts[-1] == 'gz' and filepath_parts[-2] == 'csv':
        with gzip.open(filepath, 'rt') as csvfile:
            csv_reader = csv.reader(csvfile)
            # Skip the header row if it exists
            next(csv_reader, None)

            load_witnesses(csv_reader,node_dict)

            # for row in csv_reader:
            #     print(row)
    else:
        print("{} does not end in .gz".format(filepath))
        return 1

# returns a list of all the node_list pickle files in <directory>
def find_node_list_pickles(directory):
    node_list_pickle_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pickle') and str(file).startswith("node_dict"):
                node_list_pickle_files.append(file)
    return node_list_pickle_files

# returns a list of all the distance pickle files in <directory>
def find_distance_pickles(directory):
    distance_pickle_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pickle') and str(file).startswith("distances-data"):
                distance_pickle_files.append(file)
    return distance_pickle_files


def build_node_dict(csvgz_file):
    node_dict = {}
    open_csv_gz_load_witnesses(csvgz_file,node_dict)
    return node_dict

# sort by timestamp
def sort_distances(all_distances):
    all_distances.sort(key=lambda x: x[5])
    return all_distances

# replaces gateway identifies with numbers to save storage
# best to pre-sort the input so the numbering scheme makes sense
# returns shortened all distances and gateway mapping
def shrink_gateways(all_distances):
    gateway_mapping = {}
    count = 0
    # two passes, first to set the mapping for tx gateway since that determines timestamp, then for rx gateway
    for index in range(len(all_distances)):
        tx_gateway = all_distances[index][-2]
        if tx_gateway not in gateway_mapping:
            count += 1
            gateway_mapping[tx_gateway] = count
        all_distances[index][-2] = gateway_mapping[tx_gateway]

    for index in range(len(all_distances)):
        rx_gateway = all_distances[index][-1]
        if rx_gateway not in gateway_mapping:
            count += 1
            gateway_mapping[rx_gateway] = count
            all_distances[index][-1] = count
        all_distances[index][-1] = gateway_mapping[rx_gateway]


    return all_distances, gateway_mapping
    # now rx_gateway

# takes as input all_distances (or shortened all distances)
# if any of TX pwr, RX pwr or a location does not exist, removes that from the list
# returns (the original) list with zero rows removed and "zero_list" containing the "rejected" rows
def remove_zero_rows_from_distances(all_distances):
    zeroes_index_list = []
    nz_index_list = []
    for index in range(len(all_distances)):
        if not index % 1000:
            print("Processing row {}".format(index))
        if 0 in all_distances[index]:
            zeroes_index_list.append(index)
        else:
            nz_index_list.append(index)
    return [all_distances[i] for i in nz_index_list], [all_distances[i] for i in zeroes_index_list]




if __name__ == "__main__":

    write_pickle_files_on = True  # flag to actually write the pickle files vs just process the data

    build_node_dict_on = False
    build_distances_list_on = False
    combine_distances_on = False
    shrink_all_distances_on = False
    remove_zero_rows_from_distances_on = False
    make_random_pickles_on = True


    # (1) ***** PICKLE node_dict of type HeliumNode *****
    # build node_dict pickle files from data_xxx.csv.gz original helium transaction data

    if build_node_dict_on:
        csvgz_files = find_csv_files(transaction_filepath)

        for csvgz_file in csvgz_files:

            file_name = csvgz_file.split("\\")[-1]
            pickle_file_name = 'node_dict-' + file_name.split('.')[0] + '.pickle'
            pickle_file_path = transaction_filepath + "\\" + pickle_file_name
            if write_pickle_files_on:
                if not os.path.exists(pickle_file_path):
                    node_dict = build_node_dict(csvgz_file)
                    with open(pickle_file_path, 'wb') as file:
                        pickle.dump(node_dict, file)
                else:
                    print("Pickle file {} already exists, skipping..".format(pickle_file_path))
            else: # just process but don't write
                node_dict = build_node_dict(csvgz_file)

    # ***** end node_dict for HeliumNodes *****

    # (2) **** PICKLE distances from node_dict HeliumNode.get_witness_distances()
    # list of [txlocation, location, distance_from_tx_location [,txpwr] [,rxpwr] [,timestamp] [,txgateway] [,wgateway]]
    # only thing missing is gain, but we can look that up separately later

    if build_distances_list_on:
        print("Building distances list")
        node_list_pickle_files = find_node_list_pickles(transaction_filepath)

        for pickle_file in node_list_pickle_files:
            pickle_file_path = transaction_filepath + "\\" + pickle_file
            print("..processing {}".format(pickle_file_path))

            new_pickle_file_name = 'distances-' + pickle_file.split('-')[-1]
            new_pickle_file_path = transaction_filepath + "\\" + new_pickle_file_name

            if write_pickle_files_on:
                if not os.path.exists(new_pickle_file_path):
                    with open(pickle_file_path, 'rb') as file:
                        node_dict = pickle.load(file)

                        distances = []
                        for node in node_dict:
                            distances += node_dict[node].get_witness_distances()




                        with open(new_pickle_file_path, 'wb') as file:
                            pickle.dump(distances,file)
                else:
                    print("Pickle file {} already exists, skipping..".format(new_pickle_file_path))

            else:  # just process, but don't write
                with open(pickle_file_path, 'rb') as file:
                    node_dict = pickle.load(file)

                    distances = []
                    for node in node_dict:
                        distances += node_dict[node].get_witness_distances()

    # (3) **** Combine pickle distances
    # list of [txlocation, location, distance_from_tx_location [,txpwr] [,rxpwr] [,timestamp] [,txgateway] [,wgateway]]
    # entire dataset

    if combine_distances_on:
        distance_pickle_files = find_distance_pickles(transaction_filepath)
        print(distance_pickle_files)

        all_distances = []

        for pickle_file in distance_pickle_files:
            pickle_file_path = transaction_filepath + "\\" + pickle_file
            print(pickle_file_path)

            with open(pickle_file_path, 'rb') as file:
                distances = pickle.load(file)

            all_distances += distances
            print(len(all_distances))

            new_pickle_file_path = transaction_filepath + "\\" + "all_distances.pickle"


        with open(new_pickle_file_path, 'wb') as file:
            pickle.dump(all_distances, file)

    if shrink_all_distances_on:
        all_distances = []
        load_pickle_file_path = transaction_filepath + "\\" + 'all_distances.pickle'

        new_pickle_file_name = "shrink_all_distances.pickle"
        new_pickle_file_path = transaction_filepath + "\\" + new_pickle_file_name

        if not os.path.exists(new_pickle_file_path):
            with open(load_pickle_file_path, 'rb') as file:
                all_distances = pickle.load(file)

        all_distances = sort_distances(all_distances)
        shrink_all_distances, mapping = shrink_gateways(all_distances)

        with open(new_pickle_file_path, 'wb') as file:
            pickle.dump(shrink_all_distances, file)

        new_pickle_file_name = "shrink_gateway_mapping.pickle"
        new_pickle_file_path = transaction_filepath + "\\" + new_pickle_file_name

        with open(new_pickle_file_path, 'wb') as file:
            pickle.dump(mapping, file)


    if remove_zero_rows_from_distances_on:
        print("Removing zero rows..")
        load_pickle_file_path = transaction_filepath + "\\" + "shrink_all_distances.pickle"

        with open(load_pickle_file_path, 'rb') as file:
            shrink_distances = pickle.load(file)

        shrink_distances, zero_list = remove_zero_rows_from_distances(shrink_distances)

        pickle_file_path = transaction_filepath + "\\shrink_distances_nz.pickle"
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(shrink_distances, file)

        pickle_file_path = transaction_filepath + "\\zero_list_distances.pickle"
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(zero_list, file)

    if make_random_pickles_on:

        if not "shrink_distances" in locals():
            load_pickle_file_path = transaction_filepath + "\\" + "shrink_distances_nz.pickle"
            with open(load_pickle_file_path, 'rb') as file:
                shrink_distances = pickle.load(file)

        n = 100
        list100 = random.sample(shrink_distances, n)
        new_pickle_file_path = transaction_filepath + "\\shrink_distances_100.pickle"
        with open(new_pickle_file_path, 'wb') as file:
            pickle.dump(list100, file)

        n = 1000
        list1000 = random.sample(shrink_distances, n)
        new_pickle_file_path = transaction_filepath + "\\shrink_distances_1000.pickle"
        with open(new_pickle_file_path, 'wb') as file:
            pickle.dump(list1000, file)

        n = 10000
        list10000 = random.sample(shrink_distances, n)
        new_pickle_file_path = transaction_filepath + "\\shrink_distances_10000.pickle"
        with open(new_pickle_file_path, 'wb') as file:
            pickle.dump(list10000, file)




    code.interact(local=locals())