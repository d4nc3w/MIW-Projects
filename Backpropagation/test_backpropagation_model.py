import pandas as pd
import numpy as np
import backpropagation_model as bm

def test_model():
    bm.prepare_data("gender/Transformed Data Set - Sheet1.csv", "test_dataset.csv")
    df = pd.read_csv("test_dataset.csv")
    X_test = df.drop("Gender", axis=1).to_numpy()
    y_test = df["Gender"].to_numpy()
    Y_test = bm.encode_with_one_hot_encoding(y_test)

    X_test = X_test.T
    Y_test = Y_test.T

    Y_pred = bm.predict(X_test, bm.network)

    y_pred_labels = np.argmax(Y_pred, axis=0)
    y_true_labels = np.argmax(Y_test, axis=0)

    accuracy = np.mean(y_pred_labels == y_true_labels)
    for i in range(X_test.shape[1]):
        print(f"Predicted = {y_pred_labels[i]}, Actual = {y_true_labels[i]}")
    print(f"Accuracy: {accuracy * 100:2f}%")

test_model()
