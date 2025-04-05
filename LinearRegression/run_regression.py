import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', required=True, help='Input data as csv file')
    parser.add_argument('-o', '--observation_window', type=int, required=True, help='Size of observation window')
    parser.add_argument('-p', '--predicted_currency', required=True, help='Predicted currency')

    args = parser.parse_args()
    if not args.data:
        raise ValueError('Cannot start without specified data csv files')
    return args

def calculate_normal_equation(x, y):
    tmp = x.T.dot(x)
    tnp = np.linalg.inv(tmp)
    tmp = tmp.dot(x.T)
    theta_hat = tmp.dot(y)
    return theta_hat

def calculate_hypothesis(x, theta):
    y_hat = theta.T.dot(x.T)
    return y_hat

def reshape_dataset(dataset: pd.DataFrame, window: int, prediction_currency: str):
    x_list = []
    y_list = []

    for i in range(len(dataset) - window):
        window_data = dataset.iloc[i:i+window]
        prediction = dataset.iloc[i+window][prediction_currency]

        features = window_data.drop(columns=[prediction_currency]).values.flatten()
        x_list.append(features)
        y_list.append(prediction)

    x = np.array(x_list)
    y = np.array(y_list).reshape(-1, 1)
    return x, y

def main(args: argparse.Namespace)-> None:
    data = pd.read_csv(args.data)
    observation_window = args.observation_window
    currency = args.predicted_currency

    X, y = reshape_dataset(data, observation_window, currency)
    X = np.c_[np.ones((X.shape[0], 1)), X]
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    theta_hat = calculate_normal_equation(X, y)
    y_hat = calculate_hypothesis(X, theta_hat).flatten()

    plt.plot(y, label='True')
    # plt.plot(y_hat, label='Predicted')
    plt.xlabel('Time step')
    plt.ylabel(args.predicted_currency)
    plt.savefig("prediction.png")
    plt.show()

if __name__ == '__main__':
    main(parse_args())



