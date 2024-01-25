

# This program parses .csv transaction files from the helium archive and displays information, but doesn't store
# transaction type "poc_receipts_v1" is the data we're looking for, it contains the RF measurements and witness information

import csv
import sys
import json
import code
import pickle
import h3

# Increase the field size limit (adjust as needed)
csv.field_size_limit(sys.maxsize)

# helium transaction class; of the form "block,hash,type,fields,time"
class HeliumTransaction:
    def __init__(self, block, hash, type, fields, time):
        self.block = block
        self.hash = hash
        self.type = type
        self.fields = json.loads(fields),  # Parse JSON-formatted text into a dictionary
        self.time = time

# stores tx/rx pairs for a node
# TODO what if a node changes owners or locations?
# for start time, do we want initialization announcement or just pull from receipt transaction?
# for a given node we can see if there are multiple initialization announcements (in case node is moved)
# TODO should we also figure out what the antenna gain is?
class HeliumNode:
    def __init__(self, gateway, owner, location):
        self.gateway = gateway
        self.owner = owner
        self.location = location # this will store initial location, a later change won't show
        self.start_time = 0 # will be updated with timestamp for first transaction on the blockchain
        self.last_time = 0 # will be updated with last observed transaction on the blockchain
        self.witness_transactions = {} # index by TX broadcast timestamp

    def add_witness_transaction(self,witness_transaction):
        self.witness_transactions[witness_transaction.timestamp] = witness_transaction

    # returns a list of witness distances and, optionally timestamps and RF values
    def get_witness_distances(self,timestamp=False,tx=False,rx=False):
        distances = [] # list of distances and related information for a transaction
        for wevent in self.witness_transactions:
            trxdistance = [] # list like [txlocation, location,
            # distance_from_tx_location [,txpwr] [,rxpwr] [,timestamp] [,txgateway] [,wgateway]]
            transaction = self.witness_transactions[wevent]
            timestamp = transaction.timestamp
            if not transaction.witness_dict:
                continue
            for gateway in transaction.witness_dict:
                witness = transaction.witness_dict[gateway]
                location = witness["location"]
                rxpwr = witness["rxpwr"]
                wgateway = witness["gateway"]
            distance_from_tx_location = h3.point_dist(h3.h3_to_geo(transaction.location),h3.h3_to_geo(location),unit='m')
            distances.append([transaction.location, location, distance_from_tx_location,
                              transaction.txpwr, rxpwr, timestamp, self.gateway, wgateway])

        return distances

# this class stores all witness records for a single tx challenge event for a node
class WitnessTransaction:
    def __init__(self,timestamp, txpwr, txlocation):
        self.timestamp = timestamp
        self.txpwr = txpwr
        self.location = txlocation # include location to account for a change in location
        self.witness_dict= {} # these are indexed by gateway rather than timestamp, witnesses should be unique
                               # for a given tx timestamp and gateway; could also use but prob unnecessary

    def add_witness(self,wtimestamp, wgateway, wlocation, rxpwr, freq, channel):
        self.witness_dict[wtimestamp] = {"gateway":wgateway,"location":wlocation,"rxpwr":rxpwr,\
                                        "freq":freq,"channel":channel}




# Function to read the CSV file and create instances of the HeliumTransaction class
def read_csv(file_path, n=-1):
    helium_transactions = []

    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip the header row if it exists
        next(csv_reader, None)

        for row in csv_reader:
            # Assuming the CSV file has five columns
            if len(row) >= 5:
                transaction = HeliumTransaction(row[0], row[1], row[2], row[3], row[4])
                helium_transactions.append(transaction)
                n -= 1  # Decrement the counter
                if n == 0:
                    break  # Break the loop after processing the desired number of rows


    return helium_transactions

# return a list of transaction types found in the file; transaction type is the third field
def get_all_transaction_types(file_path):
    transaction_types = {}

    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip the header row if it exists
        next(csv_reader, None)

        for row in csv_reader:
            # Assuming the CSV file has five columns
            if len(row) >= 5:
                transaction_type = row[2]

                if not transaction_type in transaction_types:
                    transaction_types[transaction_type] = 1
                else:
                    transaction_types[transaction_type] += 1

                if transaction_type == "poc_receipts_v1":
                    fields = json.loads(row[3])
                    print(fields)
                    if "receipt" in fields["path"][0] and fields["path"][0]["receipt"]:
                        tmp_receipt = fields["path"][0]["receipt"]
                        # gateway is gateway address or unique identifier for a node..
                        print("@{} {} {} {} {}".format(tmp_receipt["timestamp"],tmp_receipt["gateway"],fields["path"][0]["challengee_owner"],fields["path"][0]["challengee_location"],tmp_receipt["tx_power"]))

                        tmp_witnesses = fields["path"][0]["witnesses"]
                        for witness in tmp_witnesses:
                            print("@{} {} {} {} {} {} {}".format(witness["timestamp"],witness["gateway"],witness["owner"],witness["location"],witness["signal"],witness["frequency"],witness["channel"],witness["snr"]))
                    # for witness in fields:
                    #     print(row[2], row[3])  # print the new transaction type and its fields

    return transaction_types

def load_witnesses(file_path, node_dict):
    # transaction_types = {}

    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip the header row if it exists
        next(csv_reader, None)

        for row in csv_reader:
            # Assuming the CSV file has five columns
            if len(row) >= 5:
                transaction_type = row[2]

                # if not transaction_type in transaction_types:
                #     transaction_types[transaction_type] = 1
                # else:
                #     transaction_types[transaction_type] += 1

                if transaction_type == "poc_receipts_v1":
                    fields = json.loads(row[3])
                    #print(fields)
                    if "receipt" in fields["path"][0] and fields["path"][0]["receipt"]:
                        tmp_receipt = fields["path"][0]["receipt"]
                        # gateway is gateway address or unique identifier for a node..
                        if tmp_receipt["gateway"] not in node_dict:
                            node_dict[tmp_receipt["gateway"]] = HeliumNode(tmp_receipt["gateway"],fields["path"][0]["challengee_owner"],fields["path"][0]["challengee_location"]) # initialize that node

                        #print("@{} {} {} {} {}".format(tmp_receipt["timestamp"],tmp_receipt["gateway"],fields["path"][0]["challengee_owner"],fields["path"][0]["challengee_location"],tmp_receipt["tx_power"]))

                        tmp_witness_transaction = WitnessTransaction(tmp_receipt["timestamp"],tmp_receipt["tx_power"],fields["path"][0]["challengee_location"])
                        for witness in fields["path"][0]["witnesses"]:
                            #print("@{} {} {} {} {} {} {}".format(witness["timestamp"],witness["gateway"],witness["owner"],witness["location"],witness["signal"],witness["frequency"],witness["channel"],witness["snr"]))
                            tmp_witness_transaction.add_witness(witness["timestamp"],witness["gateway"],witness["location"],witness["signal"],witness["frequency"],witness["channel"])

                        node_dict[tmp_receipt["gateway"]].add_witness_transaction(tmp_witness_transaction)
                    # for witness in fields:
                    #     print(row[2], row[3])  # print the new transaction type and its fields

    #return transaction_types

# Example usage
if __name__ == "__main__":
    file_path = 'data_xcx.csv'  # Replace with the path to your CSV file
    n_rows_to_process = 100  # Specify the number of rows to process
    node_dict = {} # we want to store nodes indexed by gateway because that will be unique
    load_witnesses(file_path,node_dict)
    #print(transaction_list)

    # store the node_dict so we don't have to re-run and wait each time!
    pickle_file = 'node_list' + file_path.split('.')[0] + '.pickle'
    with open(pickle_file, 'wb') as file:
        pickle.dump(node_dict,file)

    # Open an interactive Python terminal
    code.interact(local=locals())