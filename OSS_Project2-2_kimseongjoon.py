### 12181583 김성준 ###
###   Project 2-2   ### 

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

def sort_dataset(dataset_df):
    sorted = dataset_df.sort_values('year', ascending=True)
    return sorted

def split_dataset(dataset_df):
    target = dataset_df.loc[:, dataset_df.columns != 'salary']
    rescale = dataset_df['salary'] * 0.001
    X_train = target[:1718]
    X_test = target[1718:]
    Y_train = rescale[:1718]
    Y_test = rescale[1718:]
    return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
    result = dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]
    return result

def train_predict_decision_tree(X_train, Y_train, X_test):
    decision_tree = DecisionTreeRegressor(random_state=0)
    decision_tree.fit(X_train, Y_train)
    pred = decision_tree.predict(X_test)
    return pred

def train_predict_random_forest(X_train, Y_train, X_test):
    random_forest = RandomForestRegressor(random_state=0)
    random_forest.fit(X_train, Y_train)
    pred = random_forest.predict(X_test)
    return pred

def train_predict_svm(X_train, Y_train, X_test):
    svm = make_pipeline(StandardScaler(), SVR())
    svm.fit(X_train, Y_train)
    return svm.predict(X_test)

def calculate_RMSE(labels, predictions):
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    return rmse

if __name__=='__main__':
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
    
    sorted_df = sort_dataset(data_df)  
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
    
    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)
    
    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))  
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))  
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
