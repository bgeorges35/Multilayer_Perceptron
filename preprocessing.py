import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def Preprocessing_predict(df, Lactivation):
    Y = np.array(df[1])
    Y = np.where(Y == 'B', 0, 1)
    X = np.array(df.iloc[:, 2:])
    X = (X - np.min(X)) / (np.max(X) - np.min(X))  # Min-Max Normalization
    X = X.T
    if (Lactivation == "sigmoid"):
        Y = Y.reshape((1, len(Y)))

    if (Lactivation == "softmax"):
        enc = OneHotEncoder(sparse=False, categories='auto')
        Y = enc.fit_transform(Y.reshape(len(Y), -1))
    return X, Y


def preprocessing_data(df, Lactivation):  # Preprocessing des Datas
    Y = np.array(df[1])
    Y = np.where(Y == 'B', 0, 1)
    X = np.array(df.iloc[:, 2:])
    X = (X - np.min(X)) / (np.max(X) - np.min(X))  # Min-Max Normalization
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    X_train = X_train.T
    X_test = X_test.T
    if (Lactivation == "sigmoid"):
        Y_train = Y_train.reshape((1, len(Y_train)))
        Y_test = Y_test.reshape((1, len(Y_test)))

    if (Lactivation == "softmax"):
        enc = OneHotEncoder(sparse=False, categories='auto')
        Y_train = enc.fit_transform(Y_train.reshape(len(Y_train), -1))
        Y_test = enc.transform(Y_test.reshape(len(Y_test), -1))
    return X_train, X_test, Y_train, Y_test
