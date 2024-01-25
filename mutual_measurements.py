import pickle
import numpy as np
import matplotlib.pyplot as plt

# analyze measurements from mutual pairs where two nodes witness one another
# [txlocation, location, distance_from_tx_location [,txpwr] [,rxpwr] [,timestamp] [,txgateway] [,wgateway]]



def count_mutual_witnesses(distances):
    mutuals = {}
    for row in distances:
        if row[-2] < row[1]:
            new_key = row[-2] + row[-1]
        else:
            new_key = row[-1] + row[-2]
        if new_key not in mutuals:
            mutuals[new_key] = 1
        else:
            mutuals[new_key] += 1

    return mutuals

if __name__ == "__main__":


    pickle_file = 'measurements_xcx.pickle'
    with open(pickle_file, 'rb') as file:
        distances = pickle.load(file)

    # for row in distances:
    #     print(row, (row[-1] > row[-2]))

    mutuals = count_mutual_witnesses(distances)
    #print(mutuals)
    values = list(mutuals.values())

    plt.hist(values, bins=10, edgecolor='black')  # You can adjust the number of bins as needed
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Values in the Dictionary')
    plt.show()