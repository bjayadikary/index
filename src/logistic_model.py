from linear_model import LinearModel
from numpy.linalg import pinv, norm

import util
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Logistic Regression
class LogisticRegression(LinearModel):
    def fit(self, x, y):
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        delta = 1
        ite = 1
        while delta > self.eps:
            x_theta = x.dot(self.theta)
            prev_theta = self.theta
            # Gradient of Negative Log Likelihood
            grad = -(1 / m) * (y - sigmoid(x_theta)).dot(x)  # ((m,) - (m,)) . (m, n)  => (n,)
            # Hessian
            hess = (1 / m) * (sigmoid(x_theta).dot(1 - sigmoid(x_theta))) * x.T.dot(x)
            # update theta with Newtons' method
            self.theta = self.theta - pinv(hess).dot(grad)
            # Negative log likelihood
            nll = -(1 / m) * (y.dot(np.log(sigmoid(x_theta))) + (1 - y).dot(np.log(1 - sigmoid(x_theta))))

            delta = norm(prev_theta - self.theta)

            print("Iteration: {} NLL: {} Theta: {}".format(ite, nll, self.theta))
            ite += 1

        print(f"Final theta : {self.theta}")

        return self.theta
    def predict(self, x, y = None):
        probabilities = sigmoid(x.dot(self.theta))

        # calculate Mean Squared Error
        if y is not None:
            y_predicted = ( probabilities >= 0.5).astype(np.int)   #np.array([1 if probability > 0.5 else 0 for probability in probabilities])
            mse = np.mean((y_predicted == y).astype(np.int)) #([1 if y_predicted_value == y_value else 0 for y_predicted_value, y_value in zip(y_predicted, y)])

            print(f"Logistic Regression performed with {mse} accuracy.")

        return probabilities

    def fit_with_gradient_ascent(self, x, y):
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        learning_rate = 0.5
        ite = 1
        delta = 1
        while delta > self.eps:
            prev_theta = self.theta
            # update theta with gradient ascent
            self.theta = self.theta + learning_rate * self.gradient_calc(x, y)
            # log likelihood
            ll = (1 / m) * (
                    y.dot(np.log(sigmoid(x.dot(self.theta)))) + (1 - y).dot(np.log(1 - sigmoid(x.dot(self.theta)))))

            delta = norm(prev_theta - self.theta)
            print("Iteration: {} LL:{} Theta: {}".format(ite, ll, self.theta))
            ite += 1

        print(f"Final theta : {self.theta}")

        return self.theta

    def gradient_calc(self, x, y):
        grad_sum = np.zeros(x.shape[1])
        for i in range(x.shape[0]):
            grad_sum = grad_sum + (y[i] - sigmoid(self.theta.T.dot(x[i]))) * x[i]

        return grad_sum / x.shape[0]


def main(train_path, valid_path, save_path, eps = 1e-5, with_gradient_ascent=False):
    x_train, y_train = util.load_data(train_path, add_intercept=True)
    if with_gradient_ascent:
        model = LogisticRegression(eps)
        lr_theta = model.fit_with_gradient_ascent(x_train, y_train)

    else:
        model = LogisticRegression(eps)
        lr_theta = model.fit(x_train, y_train)

    # Accuracy on validation set
    x_valid, y_valid = util.load_data(valid_path, add_intercept=True)
    y_predicted = model.predict(x_valid, y_valid)

    # plotting data and decision boundary
    util.plot_data(x_train, y_train, lr_theta, save_path)
    np.savetxt(save_path + '.txt', y_predicted)

    return lr_theta


if __name__ == '__main__':
    train_pathA = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds1_train.csv"
    valid_pathA = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds1_valid.csv"
    save_path = "C:\\Users\\Jay\\PycharmProjects\\ps1\\src\\output\\p01b_pred_1"
    main(train_pathA, valid_pathA, save_path)
