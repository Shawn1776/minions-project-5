import numpy as np 
import pandas as pd 
from subprocess import check_output
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
    
def trainSetOverview():
    global train
    print ("Training data overview \n")
    print ("Column Headers:", list(train.columns.values), "\n")
    print (train.dtypes)
    for col in train:
        unique = train[col].unique()
        print ('\n' + str(col) + ' has ' + str(unique.size) + ' unique values')
        if (True in pd.isnull(unique)):
            print (str(col) + ' has ' + str(pd.isnull(train[col]).sum()) + ' missing values \n')

def processData():
    global train, test
    print ("Processing the datasets.. \n")
    for data in [train,test]:
        for i in range(1,11):
            data['char_'+str(i)].fillna('type 0', inplace = 'true')
            data['char_'+str(i)] = data['char_'+str(i)].str.lstrip('type ').astype(np.int32)
        
    data['activity_category'] = data['activity_category'].str.lstrip('type ').astype(np.int32)
    
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data.drop('date', axis=1, inplace=True)
    
    for i in range(1,10):
        people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    for i in range(10, 38):
        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)
    
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    people['year'] = people['date'].dt.year
    people['month'] = people['date'].dt.month
    people['day'] = people['date'].dt.day
    people.drop('date', axis=1, inplace=True)
    
def merge():
    global train, test, people
    print ("Merging the datasets.. \n")

    train = pd.merge(train, people, how='left', on='people_id', left_index=True)
    train.fillna(-1, inplace=True)
    test = pd.merge(test, people, how='left', on='people_id', left_index=True)
    test.fillna(-1, inplace=True)

    train = train.drop(['people_id'], axis=1)
    
def featureRanking():
    model = LogisticRegression()
    rfe = RFE(model, 28)
    fit = rfe.fit(X, Y)
    print("Num Features: %d") % fit.n_features_
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_
    
    X = X.drop(['char_1_x'], axis=1)
    X = X.drop(['char_3_x'], axis=1)
    X = X.drop(['char_4_x'], axis=1)
    X = X.drop(['char_5_x'], axis=1)
    X = X.drop(['char_9_x'], axis=1)
    X = X.drop(['char_10_x'], axis=1)
    X = X.drop(['day_x'], axis=1)
    X = X.drop(['day_y'], axis=1)
    X = X.drop(['char_31'] , axis=1)
    X = X.drop(['char_29'], axis=1)

    model(X,Y)
    
def model(X,Y):
    global train,test
    rfc = RandomForestClassifier(n_estimators=96)
    scores = cross_val_score(rfc, X, Y, cv=4)
    print ("Mean accuracy of Random Forest: " + scores.mean())
    rfc = rfc.fit(X, Y)
    
    test = test.drop(['people_id'], axis=1)
    test_x = test.iloc[:, 1:]
    predictions = model.predict(valData.map(lambda x: x.features))
    test['outcome'] = predictions
    test[['activity_id', 'outcome']].to_csv('submission.csv', index=False)
    
def main():
    trainSetOverview()
    processData()
    merge()
    featureRanking()

print ("Loading input files.. \n")
people = pd.read_csv(sys.argv[1], dtype={'people_id': np.str, 'activity_id': np.str, 'char_38': np.int32}, parse_dates=['date'])
train = pd.read_csv(sys.argv[2], dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8}, parse_dates=['date'])
test = pd.read_csv(sys.argv[3], dtype={'people_id': np.str, 'activity_id': np.str}, parse_dates=['date'])
    
main()
