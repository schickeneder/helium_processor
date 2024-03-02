import pickle
import numpy as np
import matplotlib.pyplot as plt

# plot an analyze distance data..
# [txlocation, location, distance_from_tx_location [,txpwr] [,rxpwr] [,timestamp] [,txgateway] [,wgateway]]


# plots based on a list of distance data (example above)
def plot_distances(distances):
    np_distances = np.array(distances)
    # for row in distances:
    #     print(row[2],row[3]-row[4])

    y_values = np_distances[:,2].astype(float)
    x_values = np_distances[:,3].astype(int)-np_distances[:,4].astype(int)

    print(distances[0][-2],distances[0][-1])

    plt.scatter(x_values[1:2000],y_values[1:2000])
    # plt.xscale('log')
    coefficients = np.polyfit(x_values,y_values,1)
    poly = np.poly1d(coefficients)
    plt.plot(x_values,poly(x_values),color='red')
    plt.show()

# plots the standard distance vs RSSI and standard deviation for each mutual node pair
def plot_distance_SD_distance(pairs_distance_info):
    pairs = list(pairs_distance_info.keys())
    x_values = [] # distance
    y_values = [] # average RSS diff
    y_SD = [] # SD of RSS diff
    for pair in pairs:
        distances = pairs_distance_info[pair]
        np_distances = np.array(distances)
        x_values.append(np_distances[:,2].astype(float).mean())
        y_values.append((np_distances[:,3].astype(int)-np_distances[:,4].astype(int)).mean())

    coefficients = np.polyfit(y_values,x_values,1)
    poly = np.poly1d(coefficients)
    plt.plot(x_values,poly(x_values),color='red')


    plt.scatter(y_values,x_values)
    plt.show()

if __name__ == "__main__":


    # pickle_file = 'measurements_xcx.pickle'
    # with open(pickle_file, 'rb') as file:
    #     distances = pickle.load(file)


    pickle_file = 'pairs_distances.pickle'
    with open(pickle_file, 'rb') as file:
        pairs_distance_info = pickle.load(file)

    pairs = list(pairs_distance_info.keys())
    # for row in pairs_distance_info[pairs[0]]:
    #     print(row)
    # print(pairs_distance_info[pairs[0]])
    # plot_distances(pairs_distance_info[pairs[0]])

    plot_distance_SD_distance(pairs_distance_info)

    # for pair in pairs:
    #     plot_distances(pairs_distance_info[pair])

    # np_distances = np.array(distances)
    # # for row in distances:
    # #     print(row[2],row[3]-row[4])
    #
    # y_values = np_distances[:,2].astype(float)
    # x_values = np_distances[:,3].astype(int)-np_distances[:,4].astype(int)
    #
    #
    # plt.scatter(x_values[1:2000],y_values[1:2000])
    # # plt.xscale('log')
    # coefficients = np.polyfit(x_values,y_values,1)
    # poly = np.poly1d(coefficients)
    # plt.plot(x_values,poly(x_values),color='red')
    # plt.show()