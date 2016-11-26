import numpy as np 
import pandas as pd 
from subprocess import check_output
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def loadData(peoplePath, trainPath, testPath):
    print ("Loading input files..")
    people = pd.read_csv(peoplePath,
                       dtype={'people_id': np.str,
                              'activity_id': np.str,
                              'char_38': np.int32},
                       parse_dates=['date'])
    train = pd.read_csv(trainPath,
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])
    test = pd.read_csv(testPath,
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])

def main():
    loadData(sys.argv[1], sys.argv[2], sys.argv[3])

main()
