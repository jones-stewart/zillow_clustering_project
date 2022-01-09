import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures


def scale_and_cluster(train, validate, test, model_vars, cluster_vars, k):
    '''
This function takes in a list of variables to be modeled on, a list of features to
    cluster by, and a k-value to determine the number of clusters to create. 
It returns train, validate, and test dataframes with a cluster column, as well as scaled versions
    of train, validate, and test with a cluster column.
    '''
    # define independent variables
    
    X_train = pd.concat([train[model_vars], train[cluster_vars]], axis=1)
    X_train = X_train.loc[:,~X_train.columns.duplicated()]
    X_validate = pd.concat([validate[model_vars], validate[cluster_vars]], axis=1)
    X_validate = X_validate.loc[:,~X_validate.columns.duplicated()]
    X_test = pd.concat([test[model_vars], test[cluster_vars]], axis=1)
    X_test = X_test.loc[:,~X_test.columns.duplicated()]
    # scale features
    scaler = StandardScaler().fit(X_train)
    X_scaled_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)
    X_scaled_validate = pd.DataFrame(scaler.transform(X_validate), columns = X_validate.columns)
    X_scaled_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
    # create kmeans object
    kmeans = KMeans(n_clusters=k, random_state=123)
    # fit object
    kmeans.fit(X_scaled_train[cluster_vars])
    # make predictions
    kmeans.predict(X_scaled_train[cluster_vars])
    # create columns for predictions
    train['cluster'] = kmeans.predict(X_scaled_train[cluster_vars])
    X_scaled_train['cluster'] = kmeans.predict(X_scaled_train[cluster_vars])
    validate['cluster'] = kmeans.predict(X_scaled_validate[cluster_vars])
    X_scaled_validate['cluster'] = kmeans.predict(X_scaled_validate[cluster_vars])
    test['cluster'] = kmeans.predict(X_scaled_test[cluster_vars])
    X_scaled_test['cluster'] = kmeans.predict(X_scaled_test[cluster_vars])
    model_vars.append('cluster')
    
    X_train = X_scaled_train[model_vars]
    X_validate = X_scaled_validate[model_vars]
    X_test = X_scaled_test[model_vars]
    
    return X_train, X_validate, X_test


def encode_clusters(X_train, X_validate, X_test):
    '''
This function takes in scaled X_train/validate/test dfs and encodes the cluster column. Dropping the 
original unencoded cluster column.
    '''
    
    #encode cluster column
    dummy_train_df = pd.get_dummies(X_train.cluster, prefix='cluster')
    dummy_validate_df = pd.get_dummies(X_validate.cluster, prefix='cluster')
    dummy_test_df = pd.get_dummies(X_test.cluster, prefix='cluster')
    
    encoded_train = pd.concat([X_train , dummy_train_df], axis = 1)
    encoded_validate = pd.concat([X_validate , dummy_validate_df], axis = 1)
    encoded_test = pd.concat([X_test , dummy_test_df], axis = 1)
    
    X_train_encoded = encoded_train.drop(columns = ['cluster'])
    X_validate_encoded = encoded_validate.drop(columns = ['cluster'])
    X_test_encoded = encoded_test.drop(columns = ['cluster'])
    
    return X_train_encoded, X_validate_encoded, X_test_encoded

def model_features_df(target, X_train, X_validate, y_train, y_validate, model_features):
#baseline_prediction
    median = y_train[target].median()
    y_train['baseline_pred'] = median
    y_validate['baseline_pred'] = median
   
    rmse_baseline_train = mean_squared_error(y_train[target], y_train.baseline_pred)**(1/2)
    rmse_baseline_validate = mean_squared_error(y_validate[target], y_validate.baseline_pred)**(1/2)

   #model01_prediction
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train[target])
   
    y_train['model01_pred'] = lm.predict(X_train)
    y_validate['model01_pred'] = lm.predict(X_validate)
   
    rsme_model01_train = (mean_squared_error(y_train.logerror, y_train['model01_pred']))**(1/2)
    rsme_model01_validate = (mean_squared_error(y_validate.logerror, y_validate['model01_pred']))**(1/2)
   
   
   #model02_prediction
    lars = LassoLars(alpha=1.0)
    lars.fit(X_train, y_train[target])
   
    y_train['model02_pred'] = lars.predict(X_train)
    y_validate['model02_pred'] = lars.predict(X_validate)
   
    rsme_model02_train = (mean_squared_error(y_train.logerror, y_train['model02_pred']))**(1/2)
    rsme_model02_validate = (mean_squared_error(y_validate.logerror, y_validate['model02_pred']))**(1/2)

    
    #model03_prediction
    lm2 = LinearRegression(normalize=True)
    lm2.fit(X_train[model_features], y_train[target])
   
    y_train['model03_pred'] = lm2.predict(X_train[model_features])
    y_validate['model03_pred'] = lm2.predict(X_validate[model_features])
   
    rsme_model03_train = (mean_squared_error(y_train.logerror, y_train['model03_pred']))**(1/2)
    rsme_model03_validate = (mean_squared_error(y_validate.logerror, y_validate['model03_pred']))**(1/2)
    
    #model04_prediction
    lars2.fit(X_train[model_features], y_train[target])
   
    y_train['model04_pred'] = lars2.predict(X_train[model_features])
    y_validate['model04_pred'] = lars2.predict(X_validate[model_features])
   
    rsme_model04_train = (mean_squared_error(y_train.logerror, y_train['model04_pred']))**(1/2)
    rsme_model04_validate = (mean_squared_error(y_validate.logerror, y_validate['model04_pred']))**(1/2)
   
    return pd.DataFrame({'OLS with Clusters':[rsme_model01_train, rsme_model01_validate, (rsme_model01_train-rsme_model01_validate)],
                     'OLS without Clusters':[rsme_model03_train, rsme_model03_validate, (rsme_model03_train-rsme_model03_validate)],
                     'LassoLars with Clusters':[rsme_model02_train, rsme_model02_validate, (rsme_model02_train-rsme_model02_validate)],
                     'LassoLars without Clusters':[rsme_model04_train, rsme_model04_validate, (rsme_model04_train-rsme_model04_validate)],
                     'Baseline':[rmse_baseline_train, rmse_baseline_validate, (rmse_baseline_train-rmse_baseline_validate)]},
                    index=['train', 'validate', 'difference'])

def model_features_df_report(X_scaled_train, X_scaled_validate, y_train, y_validate):
#baseline_prediction
    median = y_train.logerror.median()
    y_train['baseline_pred'] = median
    y_validate['baseline_pred'] = median
   
    rmse_baseline_train = mean_squared_error(y_train.logerror, y_train.baseline_pred)**(0.5)
    rmse_baseline_validate = mean_squared_error(y_validate.logerror, y_validate.baseline_pred)**(0.5)

   #model01_prediction
    lm = LinearRegression(normalize=True)
    lm.fit(X_scaled_train, y_train.logerror)
   
    y_train['model01_pred'] = lm.predict(X_scaled_train)
    y_validate['model01_pred'] = lm.predict(X_scaled_validate)
   
    rsme_model01_train = (mean_squared_error(y_train.logerror, y_train['model01_pred']))**(0.5)
    rsme_model01_validate = (mean_squared_error(y_validate.logerror, y_validate['model01_pred']))**(0.5)
   
    #model02_prediction
    lars = LassoLars(alpha=1.0)
    lars.fit(X_scaled_train, y_train.logerror)
   
    y_train['model02_pred'] = lars.predict(X_scaled_train)
    y_validate['model02_pred'] = lars.predict(X_scaled_validate)
   
    rsme_model02_train = (mean_squared_error(y_train.logerror, y_train['model02_pred']))**(0.5)
    rsme_model02_validate = (mean_squared_error(y_validate.logerror, y_validate['model02_pred']))**(0.5)

    
    #model03_prediction
    lm = LinearRegression(normalize=True)
    lm.fit(X_scaled_train[['age', 'sq_ft', 'tax_value']], y_train.logerror)
   
    y_train['model03_pred'] = lm.predict(X_scaled_train[['age', 'sq_ft', 'tax_value']])
    y_validate['model03_pred'] = lm.predict(X_scaled_validate[['age', 'sq_ft', 'tax_value']])
   
    rsme_model03_train = (mean_squared_error(y_train.logerror, y_train['model03_pred']))**(0.5)
    rsme_model03_validate = (mean_squared_error(y_validate.logerror, y_validate['model03_pred']))**(0.5)
    
    #model04_prediction
    lars.fit(X_scaled_train[['age', 'sq_ft', 'tax_value']], y_train.logerror)
   
    y_train['model04_pred'] = lars.predict(X_scaled_train[['age', 'sq_ft', 'tax_value']])
    y_validate['model04_pred'] = lars.predict(X_scaled_validate[['age', 'sq_ft', 'tax_value']])
   
    rsme_model04_train = (mean_squared_error(y_train.logerror, y_train['model04_pred']))**(0.5)
    rsme_model04_validate = (mean_squared_error(y_validate.logerror, y_validate['model04_pred']))**(0.5)
   
    return pd.DataFrame({'OLS with Clusters':[rsme_model01_train, rsme_model01_validate, (rsme_model01_train-rsme_model01_validate)],
                     'OLS without Clusters':[rsme_model03_train, rsme_model03_validate, (rsme_model03_train-rsme_model03_validate)],
                     'LassoLars with Clusters':[rsme_model02_train, rsme_model02_validate, (rsme_model02_train-rsme_model02_validate)],
                     'LassoLars without Clusters':[rsme_model04_train, rsme_model04_validate, (rsme_model04_train-rsme_model04_validate)],
                     'Baseline':[rmse_baseline_train, rmse_baseline_validate, (rmse_baseline_train-rmse_baseline_validate)]},
                    index=['train', 'validate', 'difference'])