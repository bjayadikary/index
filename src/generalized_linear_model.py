import numpy as np
import util
from numpy.linalg import norm, pinv

from linear_model import LinearModel


class GLM(LinearModel):
    def fit(self, x, y, family, link_fn, with_gradient=False):
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        ite = 1
        self.eps = 1e-3
        self.max_iter = 100000
        self.learning_rate = 1e-7
        delta = 1
        for i in range(self.max_iter):
            x_theta = x.dot(self.theta)
            prev_theta = self.theta

            # Log likelihood of GLM => scalar value
            ll = (1 / m) * (np.log(y) + y.dot(x.dot(self.theta)) - self.log_partition_fn(x.dot(self.theta), family)).sum()

            # Gradient of Log likelihood of GLM => (n, )
            grad = (1 / m) * (y - self.response_fn(x_theta, link_fn)).dot(x)

            # Hessian => (n, n)
            hess = -(1 / m) * self.response_fn(x_theta, link_fn).dot(1 - self.response_fn(x_theta, link_fn)) * x.T.dot(
                x)

            if with_gradient:
                # Update with gradient ascent
                self.theta = prev_theta + self.learning_rate * grad
            else:
                # Update with newton's method (with Maximizing ll)
                self.theta = prev_theta - pinv(hess).dot(grad)

            print(f"Iteration: {ite} Theta: {self.theta}")
            ite += 1
            if norm(self.theta - prev_theta, ord=1) < self.eps:
                break

    def response_fn(self, x, link_fn):
        if link_fn == 'Log':
            return np.exp(x)
        elif link_fn == 'Logit':
            return 1 / (1 + np.exp(-x))
        else:
            return x

    def log_partition_fn(self, x, family):
        if family == 'Poisson':
            return np.exp(x)
        elif family == 'Bernoulli':
            return np.log(1 + np.exp(x))
        else:
            return x ** 2 / 2


def main(train_path, with_gradient=False):
    x_train, y_train = util.load_data(train_path, add_intercept=True)
    model = GLM()
    model.fit(x_train, y_train, family='Poisson', link_fn='Log', with_gradient=with_gradient)


if __name__ == "__main__":
    train_path_logistic = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds2_train.csv"
    train_path_poi = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\cs229-ps-2018-master\\ps1\\data\\ds4_train.csv"
    main(train_path_poi, with_gradient=True)
