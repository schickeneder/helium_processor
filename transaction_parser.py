

# This program parses .csv transaction files from the helium archive and stores them in a HeliumTransaction class
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
def read_csv(file_path, n):
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

# Example usage
if __name__ == "__main__":
    file_path = 'data_xcx.csv'  # Replace with the path to your CSV file
    n_rows_to_process = 1000  # Specify the number of rows to process
    helium_transactions = read_csv(file_path,n_rows_to_process)


    # Print the data in the created HeliumTransaction class instances
    for transaction in helium_transactions:
        print(f"Block: {transaction.block}, Hash: {transaction.hash}, Type: {transaction.type}, Fields: {transaction.fields}, Time: {transaction.time}")

    # Open an interactive Python terminal
    code.interact(local=locals())