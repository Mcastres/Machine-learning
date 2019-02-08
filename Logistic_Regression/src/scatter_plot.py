import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import sys

enumerate 

def main():
    if len(sys.argv) != 2:
        print("Please provide just one dataset")
        exit(0)
    # feature = int(sys.argv[2])
    # for i in range(6, 19):
    i = 6
    dataset = pd.read_csv(sys.argv[1])
    x = dataset.iloc[:, [1, i, i + 1]].values
    # y = dataset.iloc[:, 1].values
    # imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    # imputer = imputer.fit(x)
    # x = imputer.transform(x)
    HogwartsHouse1 = [[],[]]
    HogwartsHouse2 = [[],[]]
    HogwartsHouse3 = [[],[]]
    HogwartsHouse4 = [[],[]]
    
    for el in x:
        if  el[0] == "Gryffindor":
            HogwartsHouse1[0].append(el[1])
            HogwartsHouse1[1].append(el[2])
        elif el[0] == "Hufflepuff":
            HogwartsHouse2[0].append(el[1])
            HogwartsHouse2[1].append(el[2])            
        elif el[0] == "Ravenclaw":
            HogwartsHouse3[0].append(el[1])
            HogwartsHouse3[1].append(el[2])
        elif el[0] == "Slytherin":
            HogwartsHouse4[0].append(el[1])
            HogwartsHouse4[1].append(el[2])
        else:
            return print("Invalid day of week")

    n_bins = 25
    # print("salut " + str(HogwartsHouse1))
    # print("ok " + str(HogwartsHouse1[1]))
    plt.scatter(HogwartsHouse1[0], HogwartsHouse1[1], alpha=0.5)
    plt.scatter(HogwartsHouse2[0], HogwartsHouse2[1], alpha=0.5)
    plt.scatter(HogwartsHouse3[0], HogwartsHouse3[1], alpha=0.5)
    plt.scatter(HogwartsHouse4[0], HogwartsHouse4[1], alpha=0.5)
    plt.title(dataset.columns.values[i] + dataset.columns.values[i+1])
    # plt.legend(loc='upper right')
    # print(i)
    plt.show()

if __name__ == "__main__":
    main()
