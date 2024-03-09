import gzip
import csv
import numpy as np
import code
import json
import pickle
import h3
import matplotlib.pyplot as plt
import time
import multiprocessing


from pickle_maker import find_csv_files

# this file is used to process and gather information across the whole blockchain.
# only reads the files, doesn't reformat or extract and save the data, only metadata about the files.

# TODO: create a list of when each gateway asserted location, along with timestamp, owner and location
# this should be a list in case a gateway is moved.. assert_location_v1 assert_location_v2,
# add_gateway_v1, transfer_hotspot_v1/v2
# we will use this to measure behavior after incentives

# transaction types we care about

# assert_location_v1 keys: fee, hash, type, nonce, owner, payer, gateway, location, staking fee
# assert_location_v2 keys: fee, gain, hash, type, nonce, owner, payer, gateway, location, elevation, staking fee
# add_gateway_v1 keys: fee, hash, type, owner, payer, gateway, staking fee
# transfer_hotspot_v1 keys:  fee, hash type, buyer, seller, gateway, buyer_nonce, amount_to_seller
# transfer_hotspost_v2 keys:  fee, hash, type, nonce, owner, gateway, new owner

# and what we want to extract -> getway_locations
# [gateway], [timestamp], [block], [location], [owner], [gain?], [elevation?]
# gateway, 1681206994, 1823309, location, owner, [gain=0], [elevation=0]  * if gain,elevation are 0 that means v1

# in a separate list we want all add_gateways and transfer_hotspots-> gateway_transfers
# [gateway], [timestamp], [block], [old_owner=0], [new_owner] * if add_gateway, no old owner so 0, new owner = buyer

# a distance row is structured like:
# [txlocation, location, distance_from_tx_location [,txpwr] [,rxpwr] [,timestamp] [,txgateway] [,wgateway]]

transaction_filepath = r'D:\blockchain-etl-export\transactions'


def dummy_function(input):
    print("returning {}".format(input))
    return input


def get_new_gateways(row):
    if len(row) >= 5:
        transaction_type = row[2]
    if "add_gateway" in transaction_type:
        print(row)


def get_block_and_times(row):
    block = row[0]
    time = row[-1]
    return ([block, time])

# returns a dictionary containing a node and all pairwise h3 distances of that node's location to other nodes
# h3dist_limit sets the threshold beyond which a pair will not be recorded
# to avoid duplication, a pairwise distance will only be recorded for <node1> < <node2>
# h3dict maps a location to gateway(s) (node(s))
def get_pairwise_distances(h3dict, locations, h3dist_limit=100):
    node_dist_pairs_dict = {}
    count = 0
    total_rows = len(locations)
    for row in locations:
        if count%100000 == 0:
            print(f"Processing row {count}/{total_rows}")
        count += 1
        origin_node = row[0]
        origin_loc = row[3]
        node_dist_pairs_dict[origin_node] = []
        h3cells = h3.k_ring(origin_loc,h3dist_limit)
        for h3cell in h3cells:
            if h3cell in h3dict:
                for remote_node in h3dict[h3cell]:
                    if remote_node < origin_node:
                        node_dist_pairs_dict[origin_node].append(h3.h3_distance(origin_loc,h3cell))

    return node_dist_pairs_dict

# single location version of the function, for use with multi below
def get_pairwise_distances_multi_helper(h3dict, row):
    #global h3dict
    global h3dist_limit
    node_dist_pairs_dict_single = {}

    #print(f"h3dict {h3dict}")

    origin_node = row[0]
    origin_loc = row[3]
    node_dist_pairs_dict_single[origin_node] = []
    h3cells = h3.k_ring(origin_loc, h3dist_limit)
    for h3cell in h3cells:
        #print(f"processing h3cell {h3cell}")
        if h3cell in h3dict:
            #print(f"h3cell {h3cell} found in dict")
            for remote_node in h3dict[h3cell]:
                if remote_node < origin_node:
                    node_dist_pairs_dict_single[origin_node].append(h3.h3_distance(origin_loc, h3cell))

    return node_dist_pairs_dict_single
def get_pairwise_distances_multi(h3dict,locations):
    node_dist_pairs_dict = {}

    with multiprocessing.Pool(processes=32) as pool:
        results = pool.starmap(get_pairwise_distances_multi_helper, [(h3dict,location) for location in locations])

    for item in results:
        node_dist_pairs_dict.update(item)

    return node_dist_pairs_dict

# creates a dictionary of h3cells based on a location list, where location list is:
# [gateway], [timestamp], [block], [location], [owner], [gain?], [elevation?]
# and dictionary is like "<h3cell1>": [node1, node2, ..]
def populate_h3cells(locations, block_limit = 0, timestamp = 0):
    h3dict = {}

    if timestamp:
        for row in locations:
            if int(row[1]) < timestamp:
                if row[3] in h3dict:
                    h3dict[row[3]].append(row[0])
                else:
                    h3dict[row[3]] = [row[0]]
    elif block_limit:
        for row in locations:
            if int(row[2]) < block_limit:
                if row[3] in h3dict:
                    h3dict[row[3]].append(row[0])
                else:
                    h3dict[row[3]] = [row[0]]
    else:
        for row in locations:
            if row[3] in h3dict:
                h3dict[row[3]].append(row[0])
            else:
                h3dict[row[3]] = [row[0]]

    return h3dict

# returns location or transfer if the row contains it
# result1 is location, result2 is transfer
# location list:
# [gateway], [timestamp], [block], [location], [owner], [gain?], [elevation?]

# transfer list
# [gateway], [timestamp], [block], [old_owner=0], [new_owner]
def get_locations_and_transfers(row):
    if len(row) >= 5:
        transaction_type = row[2]
    if "assert_location" in transaction_type:
        # print(row)
        fields = json.loads(row[3])
        if "gain" in fields:
            gain = fields["gain"]
        else:
            gain = 0
        if "elevation" in fields:
            elevation = fields["elevation"]
        else:
            elevation = 0
        new_row = [fields["gateway"], row[4], row[0], fields["location"], fields["owner"], gain, elevation]
        # [gateway], [timestamp], [block], [location], [owner], [gain?], [elevation?]
        # print(new_row)
        return new_row, 0
    if "add_gateway" in transaction_type or "transfer_hotspot" in transaction_type:
        # print(row)
        fields = json.loads(row[3])
        if "seller" in fields:
            owner = fields["seller"]
        else:
            owner = fields["owner"]
        if "buyer" in fields:
            new_owner = fields["buyer"]
        elif "new_owner" in fields:
            new_owner = fields["new_owner"]
        else:
            new_owner = 0
        new_row = [fields["gateway"], row[4], row[0], owner, new_owner]
        # [gateway], [timestamp], [block], [old_owner=0], [new_owner]

        # print(new_row)
        return 0, new_row
    return 0, 0


# iterates through rows of a csv file and applies processor_function
# processor function returns the result some data element or 0
def process_csv_gz(filepath, processor_function):
    filepath_parts = filepath.split('.')
    csv.field_size_limit(2 ** 30)  # threw error because exceeded size and sys.maxsize is 2*64-1, also way too big
    if filepath_parts[-1] == 'gz' and filepath_parts[-2] == 'csv':
        results_list = []
        with gzip.open(filepath, 'rt') as csvfile:
            csv_reader = csv.reader(csvfile)
            # Skip the header row if it exists
            next(csv_reader, None)

            for row in csv_reader:
                result = processor_function(row)
                if result:  # only if the processor function returned something
                    results_list.append(result)

            return results_list

    else:
        print("{} does not end in .gz".format(filepath))
        return 1


# same as above, but returns two results lists
def process_csv_gz2(filepath, processor_function):
    filepath_parts = filepath.split('.')
    csv.field_size_limit(2 ** 30)  # threw error because exceeded size and sys.maxsize is 2*64-1, also way too big
    if filepath_parts[-1] == 'gz' and filepath_parts[-2] == 'csv':
        results_list1 = []
        results_list2 = []
        with gzip.open(filepath, 'rt') as csvfile:
            csv_reader = csv.reader(csvfile)
            # Skip the header row if it exists
            next(csv_reader, None)

            for row in csv_reader:
                result1, result2 = processor_function(row)
                if result1:  # only if the processor function returned something
                    results_list1.append(result1)
                if result2:
                    results_list2.append(result2)

            return results_list1, results_list2

    else:
        print("{} does not end in .gz".format(filepath))
        return 1


def get_min_max_2col(list):
    array = np.array(list)
    min_column_indices = np.argmin(array, axis=0)
    max_column_indices = np.argmax(array, axis=0)
    return (array[min_column_indices[0]].tolist(), array[min_column_indices[1]].tolist(),
            array[max_column_indices[0]].tolist(), array[max_column_indices[1]].tolist())

# get histogram of
def get_witness_h3distance_frequency_distribution(distances):
    h3_dist_dict = {}


    for row in distances:
        try:
            h3_dist = h3.h3_distance(row[0],row[1])
        except:
            print(f"Could not compute dist for {row[0]} {row[1]}")
            continue
        if h3_dist not in h3_dist_dict:
            h3_dist_dict[h3_dist] = 1
        else:
            h3_dist_dict[h3_dist] += 1

    return h3_dist_dict

def plot_hist(freq_dist_data):
    samples = list(freq_dist_data.keys())
    frequencies = list(freq_dist_data.values())

    plt.bar(samples, frequencies, color='skyblue')
    plt.show()

# receives a distances list (last two columns are tx and w gateway addresses)
#   returns a new list with the denylist elements removed
# make sure input_list and denylist are both either shrink or regular
def denylist_filter_distances_list(input_list, denylist):
    new_list = []

    for element in input_list:
        txgateway = element[-2]
        wgateway = element[-1]

        if txgateway in denylist or wgateway in denylist:
            continue
        else:
            new_list.append(element)

    return new_list

def update_h3dict(new_dict):
    global h3dict
    h3dict = new_dict


block_limit = 100000
h3dist_limit = 30
h3dict = {}

if __name__ == "__main__":

    # csvgz_files = find_csv_files(transaction_filepath)
    #
    # final_result1 = []
    # final_result2 = []
    # for csvgz_file in csvgz_files:
    #
    #
    #
    #     print("Processing file {}".format(csvgz_file))
    #
    #     result1, result2 = process_csv_gz2(csvgz_file, get_locations_and_transfers)
    #     final_result1 += result1
    #     final_result2 += result2
    #         # print(list(get_min_max_2col(result)))

        # file_name = csvgz_file.split("\\")[-1]
        # pickle_file_name = 'node_dict-' + file_name.split('.')[0] + '.pickle'
        # pickle_file_path = transaction_filepath + "\\" + pickle_file_name
        # if write_pickle_files_on:
        #     if not os.path.exists(pickle_file_path):
        #         node_dict = build_node_dict(csvgz_file)
        #         with open(pickle_file_path, 'wb') as file:
        #             pickle.dump(node_dict, file)
        #     else:
        #         print("Pickle file {} already exists, skipping..".format(pickle_file_path))
        # else: # just process but don't write
        #     node_dict = build_node_dict(csvgz_file)

    # distances_filepath = transaction_filepath + "\shrink_distances_10000.pickle"
    # with open(distances_filepath, 'rb') as file:
    #             distances = pickle.load(file)
    #
    # with open(r'D:\blockchain-etl-export\transactions\shrink_gateway_mapping.pickle', 'rb') as file:
    #             shrink_denylist = pickle.load(file)
    #
    # new_distances = denylist_filter_distances_list(distances, shrink_denylist)
    #
    # results = get_witness_h3distance_frequency_distribution(new_distances)
    #
    # #print(f"Average count {np.mean(np.array(list(results.values())))}, stdev {np.std(np.array(list(results.values)))}")
    # plot_hist(results)


    start_time = time.time()

    locations_filepath = transaction_filepath + "\locations.pickle"
    with open(locations_filepath, 'rb') as file:
                locations = pickle.load(file)

    h3dict = populate_h3cells(locations, block_limit=block_limit)
    update_h3dict(h3dict)

    #results = get_pairwise_distances(h3dict,locations, h3dist_limit)
    results = get_pairwise_distances_multi(h3dict, locations)

    print(f"Elapsed time: {(time.time()-start_time):.2f}s with block_limit {block_limit} and h3dist_limit {h3dist_limit}")
    # print(len(h3cells))

    # this is REALLY slow.. can we fix it with matrix multiplication? numpy?

    code.interact(local=locals())
