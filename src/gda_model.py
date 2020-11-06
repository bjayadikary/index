import numpy as np

from linear_model import LinearModel
from numpy.linalg import pinv
import util


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GDA(LinearModel):
    def fit(self, x, y):
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n + 1)
        x_with_y0 = x[y == 0]
        x_with_y1 = x[y == 1]
        mu0 = np.sum(x_with_y0, axis=0) / np.sum([y == 0])
        mu1 = np.sum(x_with_y1, axis=0) / np.sum([y == 1])
        phi = np.sum([y == 1]) / m
        diff = x.copy()
        diff[y == 0] -= mu0
        diff[y == 1] -= mu1
        sigma = diff.T.dot(diff) / m

        # update theta
        self.theta[1:] = (mu1 - mu0).T.dot(pinv(sigma))
        self.theta[0] = (1 / 2) * (mu0.T.dot(pinv(sigma)).dot(mu0) - mu1.T.dot(pinv(sigma)).dot(mu1)) - np.log(
            (1 - phi) / phi)

        return self.theta

    def predict(self, x, y=None):
        x = util.add_intercept_fn(x)
        prob = sigmoid(x.dot(self.theta))
        if y is not None:
            y_pred = (prob >= 0.5).astype(np.int)
            # calculate Mean Squared Error
            mse = ((y_pred == y).astype(np.int)).mean()

            print(f"GDA performed with {mse} accuracy.")
        return prob


def main(train_path, valid_path, save_path):
    x_train, y_train = util.load_data(train_path, add_intercept=False)
    model = GDA()
    model.fit(x_train, y_train)

    # Accuracy on validation set
    x_valid, y_valid = util.load_data(valid_path, add_intercept=False)
    y_predicted = model.predict(x_valid, y_valid)

    print(model.theta)

    # plotting data and decision boundary
    util.plot_data(util.add_intercept_fn(x_train), y_train, model.theta, save_path)
    np.savetxt(save_path + '.txt', y_predicted)

if __name__ == '__main__':
    train_pathA = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds1_train.csv"
    valid_pathA = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds1_valid.csv"
    train_pathB = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds2_train.csv"
    valid_pathB = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds2_valid.csv"
    save_path = "C:\\Users\\Jay\\PycharmProjects\\ps1\\src\\output\\p01e_pred_1"
    main(train_pathB, valid_pathB, save_path)



