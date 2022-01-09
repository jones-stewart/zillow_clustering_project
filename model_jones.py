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
    dummy_train_df = pd.get_dummies(X_train.cluster, prefix='cluster_')
    dummy_validate_df = pd.get_dummies(X_validate.cluster, prefix='cluster_')
    dummy_test_df = pd.get_dummies(X_test.cluster, prefix='cluster_')
    
    encoded_train = pd.concat([X_train , dummy_train_df], axis = 1)
    encoded_validate = pd.concat([X_validate , dummy_validate_df], axis = 1)
    encoded_test = pd.concat([X_test , dummy_test_df], axis = 1)
    
    X_train_encoded = encoded_train.drop(columns = ['cluster'])
    X_validate_encoded = encoded_validate.drop(columns = ['cluster'])
    X_test_encoded = encoded_test.drop(columns = ['cluster'])
    
    return X_train_encoded, X_validate_encoded, X_test_encoded



























#def create_cluster(train, validate, test, X_clusters, k):
#    '''
#THIS FUNCTION SCALES THE VARIABLES YOU WANT TO CLUSTER ON AND RETURNS A DF WITH THE CLUSTER PREDICTIONS.
#    
#INPUTS:
#    - train, validate, test: cleaned, split data
#    - X_clusters: list of unscaled features you want to cluster on
#    - k: no of clusters
#
#OUTPUTS:
#    - train, validate, test: df's with cluster predictions columns added
#    - X_clusters: df of scaled features you want to cluster on 
#    - scaler: scaler object
#    - kmeans: clustering object
#    '''
#    
#    scaler = StandardScaler(copy=True).fit(train[X_clusters])
##    X_clusters = pd.DataFrame(scaler.transform(X_clusters), columns=X_clusters.values).set_index([X_clusters.index.values])
#    
#    kmeans = KMeans(n_clusters = k, random_state = 42)
#    kmeans.fit(train[X_clusters])
#    
##    kmeans.predict(X_clusters)
#    train['cluster'] = kmeans.predict(train[X_clusters])
#    train['cluster'] = 'cluster_' + train.cluster.astype(str)
#    
#    validate['cluster'] = kmeans.predict(validate[X_clusters])
#    validate['cluster'] = 'cluster_' + validate.cluster.astype(str)
#    
#    test['cluster'] = kmeans.predict(test[X_clusters])
#    test['cluster'] = 'cluster_' + test.cluster.astype(str)
#    
#    return train, validate, test



#def scale_features(scaler, X_train, X_validate, X_test):
#    '''
#    
#    '''
#    scaler = StandardScaler(copy=True).fit(X_train)
#    X_scaled_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns + '_scaled')
#    X_scaled_validate = pd.DataFrame(scaler.transform(X_validate), columns = X_validate.columns + '_scaled')
#    X_scaled_test = pd.DataFrame(scaler.transform(X_validate), columns = X_test.columns + '_scaled')
#    
#    return X_scaled_train, X_scaled_validate, X_scaled_test





#def create_scatter_plot(x, y, df, kmeans, X_scaled, scaler, k):
#    '''
#THIS FUNCTION PLOTS A LABELED SCATTERPLOT OF TWO VARS, WITH COLOR CODED CLUSTERS. 
#
#IF YOU PLOT THE VARS YOU CLUSTERED ON, YOU WILL SEE THE CLUSTER DIVISIONS. 
#
#INPUTS: 
#    - x: x var for scatterplot
#    - y: y var for scatterplot
#    - df: train df w/cluster column
#        - df: come back and use this function on validate and test to get cluster columns for modeling!
#    - X_scaled: df of scaled features you want to cluster on
#    - scaler:: scaler object
#    - k: no of clusters
#OUTPUT: 
#    - viz: labeled scatterplot of x against y, with hue = cluster
#    '''
#    
#    plt.figure(figsize=(18, 10))
#    sns.scatterplot(x = x, y = y, data = df, hue = 'cluster')
#    
##    THESE LINES OF CODE GIVING ERROR W/X VAR, THEY PLOT THE CENTROIDS | WILL FIX LATER
#    #centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns= X_scaled.columns)
#    #sns.scatterplot(x, y, ax = plt.gca(), alpha = .30, s = 500, c = 'black')
#    #centroids.plot.scatter(x = x, y = y, ax = plt.gca(), alpha = .30, s = 500, c = 'black')
#    
#    plt.title(f'{x.upper()} and {y.upper()}, Clustered by {X_scaled.columns} | k = {k}')
#    plt.show();
#    
#def X_y_split(train, validate, test, model_features, target):
#    '''
#    
#    '''
#    #encode cluster column
#    dummy_train_df = pd.get_dummies(train.cluster)
#    dummy_validate_df = pd.get_dummies(validate.cluster)
#    dummy_test_df = pd.get_dummies(test.cluster)
#    
#    encoded_train = pd.concat([train , dummy_train_df], axis = 1)
#    encoded_validate = pd.concat([validate , dummy_validate_df], axis = 1)
#    encoded_test = pd.concat([test , dummy_test_df], axis = 1)
#    
#    #X and y splits
#    X_train = train[model_features]
#    y_train = pd.DataFrame(train[target]).reset_index(drop = 'index')
#    
#    X_validate = validate[model_features]
#    y_validate = pd.DataFrame(validate[target]).reset_index(drop = 'index')
#    
#    X_test = test[model_features]
#    y_test = pd.DataFrame(test[target]).reset_index(drop = 'index')
#    
#    #scale X
#    
#    
#    return encoded_train, encoded_validate, encoded_test, X_train, y_train, X_validate, y_validate, X_test, y_test
#

#
#
#def model_features(target, X_scaled_train, X_scaled_validate, X_scaled_test, y_train, y_validate, y_test):
#    '''
#    
#    '''
#    
#    #baseline_prediction
#    median = y_train[target].median()
#    y_train['baseline_pred'] = median
#    y_validate['baseline_pred'] = median
#    
#    rmse_baseline_train = mean_squared_error(y_train[target], y_train.baseline_pred)**(1/2)
#    rmse_baseline_validate = mean_squared_error(y_train[target], y_train.baseline_pred)**(1/2)
#    
#    print('Baseline RSME')
#    print(f'rmse_baseline_train: {rmse_baseline_train}')
#    print(f'rmse_baseline_validate: {rmse_baseline_validate}')
#    print()
#    
##    plt.hist(y_train[target], color='blue', alpha=.5, label=f'{target.upper()}')
##    plt.hist(y_train['baseline_pred'], bins=1, color='red', alpha=.5, rwidth=100, label=f'Predicted {target.upper()}')
##    plt.xlabel(f'{target.upper()}')
##    plt.legend()
##    plt.show();
#    
#    #model01_prediction
#    lm = LinearRegression(normalize=True)
#    lm.fit(X_scaled_train, y_train[target])
#    
#    y_train['model01_pred'] = lm.predict(X_scaled_train)
#    y_validate['model01_pred'] = lm.predict(X_scaled_validate)
#    
#    rsme_model01_train = (mean_squared_error(y_train.logerror, y_train['model01_pred']))**(1/2)
#    rsme_model01_validate = (mean_squared_error(y_validate.logerror, y_validate['model01_pred']))**(1/2)
#    
#    print('Model 01 | Linear Reg OLS RMSE')
#    print(f'rmse_model01_train: {rsme_model01_train}')
#    print(f'rmse_model01_validate: {rsme_model01_validate}')   
#    print()
#    
#    
#    #model02_prediction
#    lars = LassoLars(alpha=1.0)
#    lars.fit(X_scaled_train, y_train[target])
#    
#    y_train['model02_pred'] = lars.predict(X_scaled_train)
#    y_validate['model02_pred'] = lars.predict(X_scaled_validate)
#    
#    rsme_model02_train = (mean_squared_error(y_train.logerror, y_train['model02_pred']))**(1/2)
#    rsme_model02_validate = (mean_squared_error(y_validate.logerror, y_validate['model02_pred']))**(1/2)
#    
#    print('Model 02 | LassoLars RMSE')
#    print(f'rmse_model02_train: {rsme_model02_train}')
#    print(f'rmse_model02_validate: {rsme_model02_validate}')   
#    print()
#    
#    #model03_prediction
#    lm = LinearRegression(normalize=True)
#    lm.fit(X_scaled_train, y_train[target])
#    
#    y_train['model03_pred'] = lm.predict(X_scaled_train)
#    y_validate['model03_pred'] = lm.predict(X_scaled_validate)
#    
##    rsme_model03_train = (mean_squared_error(y_train.logerror, y_train['model03_pred']))**(1/2)
##    rsme_model03_validate = (mean_squared_error(y_validate.logerror, y_validate['model03_pred']))**(1/2)
##    
##    print('Model 03 | Linear Reg OLS RMSE')
##    print(f'rmse_model03_train: {rsme_model03_train}')
##    print(f'rmse_model03_validate: {rsme_model03_validate}')   
##    print()
##    
##    #model04_prediction
##    lars = LassoLars(alpha=1.0)
##    lars.fit(X_scaled_train, y_train[target])
##    
##    y_train['model04_pred'] = lars.predict(X_scaled_train)
##    y_validate['model04_pred'] = lars.predict(X_scaled_validate)
##    
##    rsme_model04_train = (mean_squared_error(y_train.logerror, y_train['model04_pred']))**(1/2)
##    rsme_model04_validate = (mean_squared_error(y_validate.logerror, y_validate['model04_pred']))**(1/2)
##    
##    print('Model 04 | LassoLars RMSE')
##    print(f'rmse_model04_train: {rsme_model04_train}')
##    print(f'rmse_model04_validate: {rsme_model04_validate}')   
##    print()
    


    
    
    
    
  
    
    
    
    

