import argparse
import pandas as pd
import sys
import pickle
from neural_network import L_layer_model, accuracy
from preprocessing import preprocessing_data, Preprocessing_predict


def parse_args():
    parser = argparse.ArgumentParser(description='Solver of N-Puzzle')
    parser.add_argument("Algorithm", type=str, choices=["train", "predict"],
                        help="choose to use train or predict")
    parser.add_argument("Dataset", type=str,
                        help="Path of dataset")
    parser.add_argument("-a", "--Activation", type=str, choices=["softmax", "sigmoid"], default=["softmax"],
                        help="Choose the activation function")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Display each epoch on training")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    try:
        df = pd.read_csv(args.Dataset, header=None)
    except Exception as e:
        sys.exit(print("{}: {}".format(type(e).__name__, e)))
    X_train, X_test, Y_train, Y_test = preprocessing_data(df, args.Activation)
    if args.Algorithm == "train":
        num_iterations = 56000
        learning_rate = 0.007

        layers_dims = [X_train.shape[0], 40, 20, 10, 5, 1] if args.Activation == "sigmoid" else [X_train.shape[0], 40, 20, 10, 5, 2]
        parameters = L_layer_model(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate, num_iterations, args.Activation, print_cost=args.verbose)
        with open('parameters.pkl', 'wb') as output:
            pickle.dump(parameters, output)

    elif args.Algorithm == "predict":
        try:
            fp = open("parameters.pkl", "rb")
        except Exception as e:
            sys.exit(print("{}: {}".format(type(e).__name__, e)))
        parameters = pickle.load(fp)
        accuracy(Y_test, X_test, parameters, args.Activation, "TestSet")
        accuracy(Y_train, X_train, parameters, args.Activation, "TrainSet")
        X, Y = Preprocessing_predict(df, args.Activation)
        accuracy(Y, X, parameters, args.Activation, "All DataSet")
##