import numpy as np
import pandas as pd

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Gradient:
    def __init__(self):
        np.random.seed(42)

    def prepare_data(self):
        admissions = pd.read_csv("binary.csv")
        # make dummy variables for rank
        data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
        data = data.drop('rank', axis=1)

        # standarize features
        for field in ['gre', 'gpa']:
            mean, std = data[field].mean(), data[field].std()
            data.loc[:, field] = (data[field] - mean)/std

        # split off random 10% of the data for testing
        sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
        data, test_data = data.ix[sample], data.drop(sample)

        # split into features and targets
        self.features, self.targets = data.drop('admit', axis=1), data['admit']
        self.features_test, self.targets_test = test_data.drop('admit', axis=1), test_data['admit']

    def run(self):
        self.prepare_data()

        n_records, n_features = self.features.shape #data points count， input fields count
        last_loss = None

        #Initialize weights
        weights = np.random.normal(scale=1/n_features**.5, size=n_features)

        # Neural Network hyperparameters
        epochs = 10000
        learnrate = 0.05

        for e in range(epochs):
            del_w = np.zeros(weights.shape)
            for x, y in zip(self.features.values, self.targets):
                output = sigmoid(x)
                error = y - output
                del_w += error * sigmoid_derivative(x) * x
            weights += learnrate * del_w / n_records

        if e % (epochs) / 10 == 0:
            out = sigmoid(np.dot(self.features, weights))
            loss = np.mean((out - self.targets) ** 2)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  Warning - Loss Increasing")
            else:
                print("Train loss: ", loss)

            last_loss = loss
        #Calculate accuracy on test data
        tes_out = sigmoid(np.dot(self.features_test, weights))
        predictions = tes_out > 0.5
        accuracy = np.mean(predictions == self.targets_test)
        print("Prediction accuracy : {:.3f}".format(accuracy))



Gradient().run()
