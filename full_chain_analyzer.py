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
from itertools import repeat
from numba import jit
import cupy as cp


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

# and what we want to extract -> gateway_locations
# [gateway], [timestamp], [block], [h3 location], [owner], [gain?], [elevation?]
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
#@jit(forceobj=True, looplift=True)
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
        #results = pool.starmap(get_pairwise_distances_multi_helper, zip(repeat(h3dict),locations))

        # zip(repeat(origin_node), h3dict[h3cell]

    for item in results:
        node_dist_pairs_dict.update(item)

    return node_dist_pairs_dict


def get_pairwise_distances_multi_helper2b(h3dict, row, h3cell):
    if h3cell in h3dict:
        for remote_node in h3dict[h3cell]:
            if remote_node < row[0]:
                return h3.h3_distance(row[3],h3cell)
def get_pairwise_distances_multi_helper2(origin_loc, remote_loc):

    if remote_loc < origin_loc:
        try:
            result = h3.h3_distance(origin_loc,remote_loc)
            if result < h3dist_limit:
                return result
        except:
            pass

    return


def get_coords_from_h3cells(locations):
    coords = []
    for row in locations:
        coords.append(h3.h3_to_geo(row[3]))

# trying this a different way..
# [gateway], [timestamp], [block], [location], [owner], [gain?], [elevation?]
# assume locations is sorted based on gateway

def get_pairwise_distances_multi2(locations):
    all_dist = []

    # for origin_node in locations:
    #     for remote_node in locations:

    with multiprocessing.Pool(processes=32) as pool:
        for location in locations:
            results = pool.starmap(get_pairwise_distances_multi_helper2, [(location[3],item[3]) for item in locations])
            all_dist += results

    return all_dist


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
    length = len(input_list)
    count = 0

    for element in input_list:
        count += 1
        if not count % 100000:
            print(f"Count {count} of {length}")
        txgateway = element[-2]
        wgateway = element[-1]

        if txgateway in denylist or wgateway in denylist:
            #print(f"removing element")
            continue

        else:
            new_list.append(element)

    return new_list

def denylist_filter_distances_list_v2(input_list, denylist):
    new_list = []
    length = len(input_list)
    count = 0

    distances = [row[2] for row in input_list]
    gateways = [[row[-2],row[-1]] for row in input_list]
    print("Done extracting gateways")
    print(f"{gateways[:10]}")

    distances_cp = cp.array(distances)
    array = cp.array(gateways)
    denylist_cp = cp.array(denylist)
    print("created cp.array, distances_cp and denylist_cp")
    print(f"{array[:10,1]}")
    print(f"{distances_cp[:10]}")
    print(f"{denylist_cp[:10]}")

    is_in_denylist = cp.any(cp.isin(array,denylist_cp), axis=1)

    print("created is_in_denylist")
    print(f"{is_in_denylist}")
    print(f"True elements {cp.sum(is_in_denylist)} and total elements {len(is_in_denylist)}")

    return list(distances_cp[~is_in_denylist])

def update_h3dict(new_dict):
    global h3dict
    h3dict = new_dict

# prints and returns percentiles for a list of values
def get_percentiles(input_values):
    # convert it!
    values = cp.array(input_values)
    percentiles = [25,50,75,90,99]
    results = []
    for percentile in percentiles:
        result = int(cp.percentile(values,percentile))
        results.append(result)

    print(f"{results}/{percentiles} are results/percentiles for {len(input_values)} rows in denylist-filtered distances list")

# returns a list like [[sum_distances1,n1,timestamp1],[sum_distances2,n2,timestamp2],..]
# where sum_distances is the sum of all pairwise distances of all nodes
# TODO: find some way of determining which nodes are active to include/add/subtract
def get_location_distance_stats(distance_threshold,as_cp_array = True):
    with open(r'D:\blockchain-etl-export\transactions\shrink2_gateway_mapping_complete.pickle', 'rb') as file:
                shrink_gateway_mapping = pickle.load(file)
    with open(r'D:\blockchain-etl-export\transactions\locations.pickle', 'rb') as file:
                locations = pickle.load(file)

    print(f"Length of locations and gateway mapping are {len(locations)} and {len(shrink_gateway_mapping)}")

    # [gateway], [timestamp], [block], [h3 location], [owner], [gain?], [elevation?]
    # convert locations (above) to  [[timestamp], [shrink_gateway], [GPS_lat,GPS_lon]],[[..
    print("creating shrink_locs")

    # count = 0
    # for element in locations:
    #     if not element[0] in shrink_gateway_mapping:
    #         #print(f"{element[0]} not in shrink_gateway_mapping")
    #         count += 1
    # print(f"{count} keys missing from shrink_gateway_mapping; it contains {len(shrink_gateway_mapping)} elements")
    # # some keys were missing but solved that now..
    shrink_locs = []
    for element in locations:
        lat,lon = h3.h3_to_geo(element[3])
        shrink_locs.append([int(element[1]), int(shrink_gateway_mapping[element[0]]), lat, lon])
    if as_cp_array:
        print("converting to cp.array")
        return cp.array(shrink_locs)
    else:
        return shrink_locs

block_limit = 100000 #1837239
h3dist_limit = 5
h3dict = {}

if __name__ == "__main__":

    #---- get locations and transfers

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

    # #---- histogram or witness distance percentiles
    #
    # distances_filepath = transaction_filepath + "\shrink_distances_nz.pickle"
    # with open(distances_filepath, 'rb') as file:
    #             distances = pickle.load(file)
    #
    # with open(r'D:\denylists\shrink_unique_denylist.pickle', 'rb') as file:
    #             shrink_denylist = pickle.load(file)
    #
    #
    # # print(f"First 10 of distances used: {distances[:10]}")
    # # #print(shrink_denylist)
    # # print(f"First 10 of shrink_denylist used: {shrink_denylist[:10]}")
    #
    #
    # # row[2] of distances contains the distances between nodes for a witness event
    # # res, pers = get_percentiles([distance[2] for distance in distances])
    # # print(f"{res}/{pers} are results/percentiles for {len(distances)} rows in full distances list")
    #
    # print("extracted pickl, starting denylist filter")
    #
    # new_distances = list(denylist_filter_distances_list_v2(distances, shrink_denylist))
    # print("Done with denylist_filter")
    # #new_distances = [distance[2] for distance in distances[not_in_deny_list]]
    # print("Done extracting just the distance columns")
    # get_percentiles(new_distances)
    #
    # #results = get_witness_h3distance_frequency_distribution(new_distances)
    #
    # #print(f"Average count {np.mean(np.array(list(results.values())))}, stdev {np.std(np.array(list(results.values)))}")
    # #plot_hist(results)


    #--------just to get length of node list-----
    # with open(r'D:\blockchain-etl-export\transactions\shrink_gateway_mapping.pickle', 'rb') as file:
    #             shrink_gateway_mapping = pickle.load(file)
    # print(f"len of shrink_gateway_mapping is {len(shrink_gateway_mapping)}")


    # #----------pairwise distances----------------
    # start_time = time.time()
    #
    # locations_filepath = transaction_filepath + "\locations.pickle"
    # with open(locations_filepath, 'rb') as file:
    #             locations = pickle.load(file)
    #
    # #locations.sort(key=lambda x: x[0])
    #
    # h3dict = populate_h3cells(locations, block_limit=block_limit)
    # # print(locations[:10])
    # # print(locations[-10:])
    # # print(len(h3dict))
    # update_h3dict(h3dict)
    # #results = get_pairwise_distances(h3dict,locations, h3dist_limit)
    # results = get_pairwise_distances_multi(h3dict, locations)
    # #results = get_pairwise_distances_multi2(locations)
    # print(f"Elapsed time: {(time.time()-start_time):.2f}s with block_limit {block_limit} and h3dist_limit {h3dist_limit}")
    # # print(len(h3cells))
    # # this is REALLY slow.. can we fix it with matrix multiplication? numpy?

    #-----------get average location distance stats-------

    results = get_location_distance_stats(distance_threshold = 50000)


    # import locations and convert gateway to shrink_gateway
    # create cp array, order by timestamp, include shrink_gateway and coord_location

    code.interact(local=locals())
