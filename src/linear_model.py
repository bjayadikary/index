class LinearModel:
    def __init__(self, learning_rate=0.2, max_iter=100, eps=1e-5, theta_0=None, verbose=True):
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        raise NotImplementedError('Subclass of Linear Model must implement fit method')

    def predict(self, x, y):
        raise NotImplementedError('Subclass of Linear Model must implement predict method')
