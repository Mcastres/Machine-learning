import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import sys
import math

def column(dataSet, i):
    return [row[i] for row in dataSet]

def Mean(data):
    return sum(data) / len(data)

def Std(data, mean):
    return math.sqrt(sum(pow(x-mean,2) for x in data) / len(data)) 

def Quartil(data, percentage):
    return sorted(data).values[int(len(data) * percentage)]

def describe(listFeatures, dataSet):
    output = [['Count'], ['Mean'], ['Std'], ['Min'], ['25%'], ['50%'], ['75%'], ['Max']]
    lenListFeatures = len(listFeatures)
    for y in range(0, lenListFeatures):
        dataColumn = sorted(column(dataSet, y))
        lenDataColumn = len(dataColumn)
        output[0].append(str(lenDataColumn))
        mean = Mean(dataColumn)
        output[1].append(str(round(mean, 6)))
        output[2].append(str(round(Std(dataColumn, mean), 6)))
        output[3].append(str(round(dataColumn[0], 6)))
        output[4].append(str(round(dataColumn[int(lenDataColumn * 0.25)], 6)))
        output[5].append(str(round(dataColumn[int(lenDataColumn * 0.5)], 6)))
        output[6].append(str(round(dataColumn[int(lenDataColumn * 0.75)], 6)))
        output[7].append(str(round(dataColumn[-1], 6)))
    output = [ np.insert(listFeatures, 0, '')] + output
    return output

def putOutput(output):
    for row in output:
        sys.stdout.write("".join(word.ljust(25) for word in row))
        sys.stdout.write("\n")

def main():
    if len(sys.argv) != 2:
        print("Please provide just one dataset")
        exit(0)

    dataset = pd.read_csv(sys.argv[1])
    x = dataset.iloc[:, 6:].values
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(x)
    x = imputer.transform(x)
    putOutput(describe(dataset.columns.values[6:], x))

if __name__ == "__main__":
    main()
