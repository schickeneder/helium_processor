import gzip
import csv
import numpy as np
import code
import json
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


# returns location or transfer if the row contains it
# result1 is location, result2 is transfer
# location list:
# [gateway], [timestamp], [block], [location], [owner], [gain?], [elevation?]

# transfer list

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


if __name__ == "__main__":

    csvgz_files = find_csv_files(transaction_filepath)

    final_result1 = []
    final_result2 = []
    for csvgz_file in csvgz_files:



        print("Processing file {}".format(csvgz_file))

        result1, result2 = process_csv_gz2(csvgz_file, get_locations_and_transfers)
        final_result1 += result1
        final_result2 += result2
            # print(list(get_min_max_2col(result)))

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

    code.interact(local=locals())
