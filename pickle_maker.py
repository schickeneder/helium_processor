import os
import gzip
import csv
import sys

from transaction_reader import load_witnesses
import code
import pickle

# Read in raw data or pickle files, process and save for quick use in other applications
# pickle filename key

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
            if file.endswith('.pickle') and "data" in str(file):
                node_list_pickle_files.append(file)
    return node_list_pickle_files

# returns a list of all the distance pickle files in <directory>
def find_distance_pickles(directory):
    distance_pickle_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pickle') and "distances-data" in str(file):
                distance_pickle_files.append(file)
    return distance_pickle_files


def build_node_dict(csvgz_file):
    node_dict = {}
    open_csv_gz_load_witnesses(csvgz_file,node_dict)
    return node_dict

if __name__ == "__main__":

    build_node_dict_on = False
    build_distances_list_on = False
    combine_distances_on = False

    # (1) ***** PICKLE node_dict of type HeliumNode *****
    # build node_dict pickle files from data_xxx.csv.gz original helium transaction data

    if build_node_dict_on:
        csvgz_files = find_csv_files(transaction_filepath)

        for csvgz_file in csvgz_files:

            file_name = csvgz_file.split("\\")[-1]
            pickle_file_name = 'node_dict-' + file_name.split('.')[0] + '.pickle'
            pickle_file_path = transaction_filepath + "\\" + pickle_file_name
            #node_dict = build_node_dict(csvgz_file)
            if not os.path.exists(pickle_file_path):
                node_dict = build_node_dict(csvgz_file)
                with open(pickle_file_path, 'wb') as file:
                    pickle.dump(node_dict, file)
            else:
                print("Pickle file {} already exists, skipping..".format(pickle_file_path))

    # ***** end node_dict for HeliumNodes *****

    # (2) **** PICKLE distances from node_dict HeliumNode.get_witness_distances()
    # list of [txlocation, location, distance_from_tx_location [,txpwr] [,rxpwr] [,timestamp] [,txgateway] [,wgateway]]
    # only thing missing is gain, but we can look that up separately later

    if build_distances_list_on:
        node_list_pickle_files = find_node_list_pickles(transaction_filepath)

        for pickle_file in node_list_pickle_files:
            pickle_file_path = transaction_filepath + "\\" + pickle_file

            new_pickle_file_name = 'distances-' + pickle_file.split('-')[-1]
            new_pickle_file_path = transaction_filepath + "\\" + new_pickle_file_name
            if not os.path.exists(new_pickle_file_path):
                with open(pickle_file_path, 'rb') as file:
                    node_dict = pickle.load(file)

                    distances = []
                    for node in node_dict:
                        distances += node_dict[node].get_witness_distances()




                    with open(new_pickle_file_path, 'wb') as file:
                        pickle.dump(distances,file)

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

    code.interact(local=locals())