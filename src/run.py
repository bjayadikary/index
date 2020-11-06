import argparse

from logistic_model import main as p01b
from gda_model import main as p01e
from positiveonly_lr_model import main as p02cde
from poisson_model import main as p03d


parser = argparse.ArgumentParser()  #argparse(.py file) with class ArgumentParser(); parser, an object of ArgumentParser();
parser.add_argument('num', nargs='?', type=int, default=0, help='Problem number to run, 0 for all problems')   #add_argument, a method from class
#Optional argument is provided with '-num' ,while positional argument doesn't come with dash(-)
args = parser.parse_args()  #parse_args() returns arguments added above to a namespace(num = 'integer value by user', and any other optional argument if included)
#Problem 1
if args.num == 0 or args.num == 1:
    p01b(train_path="C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds1_train.csv",
         valid_path="C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds1_valid.csv",
         save_path="C:\\Users\\Jay\\PycharmProjects\\ps1\\src\\output\\p01b_pred_1")
    p01e(train_path="C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds1_train.csv",
         valid_path="C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds1_valid.csv",
         save_path="C:\\Users\\Jay\\PycharmProjects\\ps1\\src\\output\\p01e_pred_1")
    p01b(train_path="C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds2_train.csv",
         valid_path="C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds2_valid.csv",
         save_path="C:\\Users\\Jay\\PycharmProjects\\ps1\\src\\output\\p01b_pred_2")
    p01e(train_path="C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds2_train.csv",
         valid_path="C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\Assignment 1\\data 2018\\ds2_valid.csv",
         save_path="C:\\Users\\Jay\\PycharmProjects\\ps1\\src\\output\\p01e_pred_2")

#Problem 2
if args.num == 0 or args.num == 2:
    p02cde(train_path = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\cs229-ps-2018-master\\ps1\\data\\ds3_train.csv",
           valid_path = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\cs229-ps-2018-master\\ps1\\data\\ds3_valid.csv",
           test_path = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\cs229-ps-2018-master\\ps1\\data\\ds3_test.csv",
           save_path = "C:\\Users\\Jay\\PycharmProjects\\ps1\\src\\output\\p02X_pred")

#Problem 3
if args.num == 0 or args.num == 3:
    p03d(train_path = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\cs229-ps-2018-master\\ps1\\data\\ds4_train.csv",
         valid_path = "C:\\Users\\Jay\\Downloads\\Documents\\AI\\Cs229\\cs229-ps-2018-master\\ps1\\data\\ds4_valid.csv",
         save_path = "C:\\Users\\Jay\\PycharmProjects\\ps1\\src\\output\\p03d_pred")