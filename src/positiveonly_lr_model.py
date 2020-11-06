import numpy as np
from logistic_model import LogisticRegression
import util

WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    save_path_c = save_path.replace(WILDCARD, 'c')
    save_path_d = save_path.replace(WILDCARD, 'd')
    save_path_e = save_path.replace(WILDCARD, 'e')

    # #Solution to C - Training on train_set with true(t) labels (ideal case)
    x_train, t_train = util.load_data(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_data(test_path, label_col='t', add_intercept=True)
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, t_train)
    t_test_predicted = model.predict(x_test,t_test)
    util.plot_data(x_test, t_test, model.theta, save_path_c)
    np.savetxt(save_path_c + '.txt', t_test_predicted)

    #Solution to D - Training on train_set with labeled(y) labels
    x_train, y_train = util.load_data(train_path, label_col='y', add_intercept=True)
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)
    t_test_predicted = model.predict(x_test, t_test)
    util.plot_data(x_test, t_test, model.theta, save_path_d)
    np.savetxt(save_path_d + '.txt', t_test_predicted)

    #Solution to E - Estimating alpha with Validation+ set (posonly) and validating on test set with true labels
    x_valid, y_valid = util.load_data(valid_path, label_col='y', add_intercept=True)
    y_valid_predicted = model.predict(x_valid)
    alpha = y_valid_predicted[y_valid ==1 ].sum() / (y_valid == 1).sum()
    model.theta[0] = model.theta[0] * alpha
    print(f'Final theta after resurrection:{model.theta}')
    util.plot_data(x_test, t_test, model.theta, save_path_e)
    t_test_predicted = model.predict(x_test, t_test)
    np.savetxt(save_path_e + '.txt', t_test_predicted)

if __name__ == '__main__':
    train_path = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\cs229-ps-2018-master\\ps1\\data\\ds3_train.csv"
    valid_path = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\cs229-ps-2018-master\\ps1\\data\\ds3_valid.csv"
    test_path = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\cs229-ps-2018-master\\ps1\\data\\ds3_test.csv"
    save_path = "C:\\Users\\Jay\\PycharmProjects\\ps1\\src\\output\\p02X_pred"
    main(train_path, valid_path, test_path, save_path)
