import sys
import pandas as pd

# Save thetas in file for the estimation
def save_thetas(theta0, theta1):
    f = open("Datasets/thetas", "w")
    text = str(theta0) + "," + str(theta1)
    f.writelines(text)

# Read the csv dataset and return an array of it
def read_csv_file(file):
    data = pd.read_csv(file)
    x = [x / 1000 for x in data.km]
    y = [x / 1000 for x in data.price]
    return [x, y]

# Caluclate the hypothesis with thetas
def hypothesis(theta0, theta1, x):
    return (theta0 + (theta1 * x))

# Execute a gradient descent
def gradient_descent(theta0, theta1, learning_rate, x, y, n):
    print("Your model is being trained...")
    while (42):
        cost_function = 0
        tmp_theta_0 = sum(hypothesis(theta0, theta1, x[i]) - y[i] for i in range(0, n)) / float(n)
        tmp_theta_1 = sum((hypothesis(theta0, theta1, x[i]) - y[i]) * x[i] for i in range(0, n)) / float(n)
        cost_function = sum(y[i] - hypothesis(theta0, theta1, x[i]) for i in range(0, n))**2 / float(n)

        tmp_theta_0 *= learning_rate
        tmp_theta_1 *= learning_rate

        if (cost_function <= 0.01):
            print("Done!")
            return (theta0 * 1000, theta1)

        theta0 -= tmp_theta_0
        theta1 -= tmp_theta_1

def main():
    theta0 = 0.0
    theta1 = 0.0
    learning_rate = 0.0001

    if len(sys.argv) != 2:
        print("Please provide just one dataset")
        exit(0)

    # Read the file
    data = read_csv_file(sys.argv[1])

    # Create len variable
    n = len(data[0])

    # Fetch and save thetas
    theta0, theta1 = gradient_descent(theta0, theta1, learning_rate, data[0], data[1], n)
    save_thetas(theta0, theta1)

if __name__== "__main__":
    main()
