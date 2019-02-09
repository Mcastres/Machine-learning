import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import Imputer
import numpy as np
import sys

def build_pair_plot(dataset):
    dataset = pd.read_csv(dataset)
    subjects = []
    d = {}
    Gryffindor_house = []
    Hufflepuff_house = []
    Ravenclaw_house = []
    Slytherin_house = []

    for i in range(6, 19):
        x = dataset.iloc[:, [1, i]].values
        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        imputer = imputer.fit(x[:, [1]])
        x[:, [1]] = imputer.transform(x[:, [1]])

        Gryffindor  = []
        Hufflepuff  = []
        Ravenclaw   = []
        Slytherin   = []

        for el in x:
            if  el[0] == "Gryffindor":
                Gryffindor.append(el[1])
            elif el[0] == "Hufflepuff":
                Hufflepuff.append(el[1])
            elif el[0] == "Ravenclaw":
                Ravenclaw.append(el[1])
            elif el[0] == "Slytherin":
                Slytherin.append(el[1])
            else:
                return print("WTF")
        subjects = dataset.columns.values[i]
        d[dataset.columns.values[i]] = Gryffindor + Hufflepuff + Ravenclaw + Slytherin
    d['Houses'] = ['Gryffindor' for s in range(0, len(Gryffindor))]  + ['Hufflepuff' for s in range(0, len(Hufflepuff))] + ['Ravenclaw' for s in range(0, len(Ravenclaw))] + ['Slytherin' for s in range(0, len(Slytherin))]
    df = pd.DataFrame(data=d)
    plot_pair(df)

def plot_pair(df):
    sns.pairplot(df, palette="husl", hue="Houses", height=2)
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Please provide just one dataset")
        exit(0)

    build_pair_plot(sys.argv[1])

if __name__ == "__main__":
    main()
