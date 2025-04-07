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
    return args

def calculate_normal_equation(x: int, y: int):
    tmp = x.T.dot(x)
    tmp = np.linalg.inv(tmp)
    tmp = tmp.dot(x.T)
    theta_hat = tmp.dot(y)
    return theta_hat

def calculate_hypothesis(x: int, theta):
    y_hat = theta.T.dot(x.T)
    return y_hat

def generate_data(dataset: pd.DataFrame, window: int, prediction_currency: str):
    feature_df = pd.DataFrame()

    for i in range(window):
        shifted = dataset.shift(-i)
        for col in dataset.columns:
            #if col != prediction_currency:
            feature_df[f'{col}_t-{window-i}'] = shifted[col]

    target = dataset[prediction_currency].shift(-window)
    feature_df['target'] = target

    feature_df.dropna(inplace=True)

    X = feature_df.drop(columns=['target']).values
    y = feature_df['target'].values.reshape(-1, 1)

    return X, y

def main(args: argparse.Namespace)-> None:
    data = pd.read_csv(args.data)
    observation_window = args.observation_window
    currency = args.predicted_currency

    X, y = generate_data(data, observation_window, currency)
    X = np.c_[np.ones((X.shape[0], 1)), X]
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    theta_hat = calculate_normal_equation(X, y)
    y_hat = calculate_hypothesis(X, theta_hat).flatten()

    plt.plot(y, label='True')
    plt.plot(y_hat, label='Predicted')
    plt.xlabel('Time step')
    plt.ylabel(args.predicted_currency)
    plt.savefig("prediction.png")
    plt.show()

if __name__ == '__main__':
    main(parse_args())

#python run_regression.py --data dataset.csv --observation_window 3 --predicted_currency Monero


