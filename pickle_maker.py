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

def build_node_dict(csvgz_file):
    node_dict = {}
    open_csv_gz_load_witnesses(csvgz_file,node_dict)
    return node_dict

if __name__ == "__main__":


    # (1) ***** PICKLE node_dict of type HeliumNode *****
    # build node_dict pickle files from data_xxx.csv.gz original helium transaction data

    csvgz_files = find_csv_files(transaction_filepath)

    for csvgz_file in csvgz_files:

        file_name = csvgz_file.split("\\")[-1]
        pickle_file_name = 'node_dict-' + file_name.split('.')[0] + '.pickle'
        pickle_file_path = transaction_filepath + "\\" + pickle_file_name
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


    #code.interact(local=locals())