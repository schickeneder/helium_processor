

# This program parses .csv transaction files from the helium archive and displays information, but doesn't store
# transaction type "poc_receipts_v1" is the data we're looking for, it contains the RF measurements and witness information

import csv
import sys
import json
import code

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
                    print(row[2],row[3][:600]) # print the new transaction type and its fields
                else:
                    transaction_types[transaction_type] += 1


    return transaction_types


# Example usage
if __name__ == "__main__":
    file_path = 'data_xcx.csv'  # Replace with the path to your CSV file
    n_rows_to_process = 1000  # Specify the number of rows to process
    transaction_list = get_all_transaction_types(file_path)
    print(transaction_list)


    # Open an interactive Python terminal
    code.interact(local=locals())