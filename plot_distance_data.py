import pickle
import numpy as np
import matplotlib.pyplot as plt

# plot an analyze distance data..
# [txlocation, location, distance_from_tx_location [,txpwr] [,rxpwr] [,timestamp]]



if __name__ == "__main__":


    pickle_file = 'measurements_xcx.pickle'
    with open(pickle_file, 'rb') as file:
        distances = pickle.load(file)

    np_distances = np.array(distances)
    # for row in distances:
    #     print(row[2],row[3]-row[4])

    y_values = np_distances[:,2].astype(float)
    x_values = np_distances[:,3].astype(int)-np_distances[:,4].astype(int)


    plt.scatter(x_values[1:2000],y_values[1:2000])
    # plt.xscale('log')
    coefficients = np.polyfit(x_values,y_values,1)
    poly = np.poly1d(coefficients)
    plt.plot(x_values,poly(x_values),color='red')
    plt.show()