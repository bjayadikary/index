import numpy as np
import util
from numpy.linalg import norm, pinv

from linear_model import LinearModel


class PoissonModel(LinearModel):
    def fit(self, x, y):
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        delta = 1
        iter = 1
        self.learning_rate = 1e-7
        while self.eps < delta:
            prev_theta = self.theta
            # Log likelihood
            ll = (1 / m) * (y.dot(x.dot(self.theta)) - self.response_fn(
                x.dot(self.theta))).sum()  # -logy! is neglected here

            # Gradient of LL
            gradient = (1 / m) * (y - self.response_fn(x.dot(self.theta))).dot(x)

            # Gradient Ascent
            self.theta = prev_theta + self.learning_rate * gradient

            # #Hessian of LL hess = -(1 / m) * (self.response_fn(x.dot(self.theta))).sum() * (x.T.dot(x)) #
            # calculating hessian and exp would give us run time error, overflow encountered, since hessian values
            # would be huge and exp will overflow so better to use either numerically stable exp or gradient ascent
            # with small learning_rate #Newton's Method self.theta = prev_theta - pinv(hess).dot(gradient)

            delta = norm(self.theta - prev_theta, ord=1)
            print(f"Iteration : {iter} Theta : {self.theta} LL : {ll}")
            iter += 1

    def predict(self, x, y=None):
        y_pred = self.response_fn(x.dot(self.theta))

        return y_pred

    def response_fn(self, x):
        return np.exp(x)


def main(train_path, valid_path, save_path):
    x_train, y_train = util.load_data(train_path, add_intercept=True)
    model = PoissonModel()
    model.fit(x_train, y_train)
    x_valid, y_valid = util.load_data(valid_path, add_intercept=True)
    y_pred = model.predict(x_valid, y_valid)
    np.savetxt(save_path + '.txt', y_pred)


if __name__ == '__main__':
    train_path = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\cs229-ps-2018-master\\ps1\\data\\ds4_train.csv"
    valid_path = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\cs229-ps-2018-master\\ps1\\data\\ds4_valid.csv"
    save_path = "C:\\Users\\Jay\\PycharmProjects\\ps1\\src\\output\\p03d_pred"
    main(train_path, valid_path, save_path)
