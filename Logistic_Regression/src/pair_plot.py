import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

def build_pair_plot(dataset):
    dataset = pd.read_csv(dataset)
    for i in range(6, 19):
        x = dataset.iloc[:, [1, i]].values

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
        plot_pair(Gryffindor, Hufflepuff, Ravenclaw, Slytherin, i, dataset)

def plot_pair(Gryffindor, Hufflepuff, Ravenclaw, Slytherin, i, dataset):
    sns.pairplot(pd.DataFrame(Gryffindor))
    sns.pairplot(pd.DataFrame(Hufflepuff))
    sns.pairplot(pd.DataFrame(Ravenclaw))
    sns.pairplot(pd.DataFrame(Slytherin))
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Please provide just one dataset")
        exit(0)

    build_pair_plot(sys.argv[1])

if __name__ == "__main__":
    main()
