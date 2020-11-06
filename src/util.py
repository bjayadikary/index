import numpy as np
import matplotlib.pyplot as plt


def load_data(data_path, label_col='y', add_intercept=False):
    # load header
    with open(file=data_path, mode='r') as fh:
        header = fh.readline().strip().split(',')

    # validate label_col
    allowed_labels = ('y', 't')
    if label_col not in allowed_labels:
        raise ValueError('Invalid label col :{} (expected {})'.format(label_col, allowed_labels))

    # load features and labels
    input_cols = [i for i in range(len(header)) if header[i].startswith('x')]
    label_col = [i for i in range(len(header)) if header[i] == label_col]
    inputs = np.loadtxt(fname=data_path, delimiter=',', skiprows=1, usecols=input_cols)
    label = np.loadtxt(fname=data_path, delimiter=',', skiprows=1, usecols=label_col)

    # if 1 dimensional input, expand dimension
    if inputs.ndim == 1:
        inputs = np.expand_dims(a=inputs, axis=-1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, label


def log_load_data(data_path, add_intercept=False):
    # load header
    with open(file=data_path, mode='r') as fh:
        header = fh.readline().strip().split(',')

    # load features and labels
    input_cols = [i for i in range(len(header)) if header[i].startswith('x')]
    label_col = [i for i in range(len(header)) if header[i].startswith('y')]
    inputs = np.loadtxt(fname=data_path, delimiter=',', skiprows=1, usecols=input_cols)
    label = np.loadtxt(fname=data_path, delimiter=',', skiprows=1, usecols=label_col)
    inputs = np.log1p(inputs)
    # if 1 dimensional input, expand dimension
    if inputs.ndim == 1:
        inputs = np.expand_dims(a=inputs, axis=-1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, label


def add_intercept_fn(inputs):
    new_inputs = np.zeros((inputs.shape[0], inputs.shape[1] + 1), dtype=inputs.dtype)
    new_inputs[:, 0] = 1
    new_inputs[:, 1:] = inputs

    return new_inputs


def plot_data(x, y, theta=None, save_path=None, correction=1.0):
    plt.figure()
    x_with_y0 = x[y == 0]
    x_with_y1 = x[y == 1]
    x1_with_y0 = x_with_y0[:, 1]
    x2_with_y0 = x_with_y0[:, 2]
    x1_with_y1 = x_with_y1[:, 1]
    x2_with_y1 = x_with_y1[:, 2]

    plt.plot(x1_with_y0, x2_with_y0, color='blue', linestyle='', marker='o')
    plt.plot(x1_with_y1, x2_with_y1, color='green', linestyle='', marker='o')

    if theta is not None:
    # for plotting decision boundary
        x1_line = np.linspace(x[:, 1].min(), x[:, 1].max())
        x2_line = (-theta[0] * correction - theta[1] * x1_line) / theta[2]
        plt.plot(x1_line, x2_line, color='red')

    plt.xlabel('x_1')
    plt.ylabel('x_2')

    if save_path is not None:
        plt.savefig(save_path)