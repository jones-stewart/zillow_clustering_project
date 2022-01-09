##########################################################################################| EXPLORE.PY

##########################################################################################| IMPORTS
import wrangle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

##########################################################################################| UNIVARIATE EXPLORE
def visualize_outliers(train):
    '''
THIS FUNCTION TAKES IN THE TRAIN DF AND PLOTS TWO ROWS OF BOXPLOTS, ALLOWING US TO VISUALIZE THE DATA 
BEFORE AND AFTER OUTLIERS REMOVED.
    '''
    
    outlier_cols = ['baths', 'beds', 'sqft', 'fullbaths', 'tax_value', 'logerror']
    raw_df = wrangle.acquire_zillow()
    clean_df = train
    
    plt.figure(figsize = (32, 6))
    
    #top row of boxplots will show data with outliers, each column will be a variable
    for i, col in enumerate(outlier_cols):
        
        plot_number = i + 1
        plt.subplot(1, len(outlier_cols), plot_number)
        plt.title(f'{col} with Outliers')
        sns.boxplot(data = raw_df, y = raw_df[col])
        plt.grid(False);
        
    plt.figure(figsize = (32, 4))
    
    #bottom row will show data with outliers removed
    for i, col in enumerate(outlier_cols):
        
        plot_number = i + 1
        plt.subplot(1, len(outlier_cols), plot_number)
        plt.title(f'{col} with Outliers Removed')
        sns.boxplot(data = clean_df, y = clean_df[col])
        plt.grid(False);
        

def var_distributions(df):
    '''
THIS FUNCTION PLOTS HISTOGRAMS FOR EACH VARIABLE IN THE DATAFRAME
THAT IS FED INTO THE FUNCTION, ALLOWING US TO SEE THE SHAPE OF THE
DATA'S DISTRIBUTION. 
    '''
    
    df.hist(figsize = (18, 9))
    plt.show()
    
##########################################################################################| MULTIVARIATE EXPLORE

def plot_vars(train):
    '''
    
    '''
    
    target = pd.DataFrame(train.columns, columns = ['target'])[7:8]
    ind_vars = pd.DataFrame(train.columns, columns = ['independent_vars']).drop([7])
    
    
    return target, ind_vars
    
def age_home(train):
    '''
    
    '''
    
    train_bins = pd.concat([train.rename(columns = {'age': 'age_'}), pd.cut(train.age, 4, labels = ['our_age', 'parents_age', 'gma_age', 'greatg_age'])], axis = 1).rename(columns = {'age': 'age_bin'})
    
    plt.figure(figsize = (12, 8))
    sns.barplot(data = train_bins, x = 'age_bin', y = 'tax_value')
    plt.title(f'Tax Values of Binned Ages')
    plt.show();
    
    t, p = stats.ttest_ind(
    train_bins[train_bins.age_bin == 'our_age'].tax_value, 
    train_bins[train_bins.age_bin == 'greatg_age'].tax_value,
    equal_var = True)
    if p > .05:
        print('Fail to Reject H(o)')
    else:
        print('Reject H(o) | There is a statistically significant difference between the means of the youngest and oldest homes.')
        
def cluster_and_scale(train, validate, test):
    '''
    
    '''

    #define Xfeatures_dfs
    X_train = train#[['age', 'tax_value', 'sqft']]
    X_validate = validate#[['age', 'tax_value', 'sqft']]
    X_test = test#[['age', 'tax_value', 'sqft']]
    
    #scale model features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
    
    #cluster and encode cluster_features
    kmeans = KMeans(n_clusters = 4, random_state=123)
    kmeans.fit(X_train[['beds', 'baths', 'sqft']])
    X_train_scaled['cluster'] = kmeans.predict(X_train[['beds', 'baths', 'sqft']])
    X_train_scaled['cluster'] = 'cluster_' + X_train_scaled.cluster.astype(str)
    
    X_validate_scaled['cluster'] = kmeans.predict(X_validate[['beds', 'baths', 'sqft']])
    X_validate_scaled['cluster'] = 'cluster_' + X_validate_scaled.cluster.astype(str)
    
    X_test_scaled['cluster'] = kmeans.predict(X_test[['beds', 'baths', 'sqft']])
    X_test_scaled['cluster'] = 'cluster_' + X_test_scaled.cluster.astype(str)
    
    #visualize clusters and model features with clusters
    plt.figure(figsize = (12, 4))
    sns.scatterplot(data = X_train_scaled, x = 'sqft', y = 'baths', size = 'beds', hue = 'cluster', alpha = 0.6)
    plt.title('Clusters Groupings Using `age`, `tax_value`, and `sqft`, k=4')
    plt.legend('')
    plt.show();
    
    plt.figure(figsize = (12, 6))
    sns.scatterplot(data = X_train_scaled, x = 'latitude', y = 'longitude', hue = 'cluster', alpha = 0.6)
    plt.title('GoePlot of clusters')
    plt.show();
    print('No Meaningful Patterns In Clusters When Plotted Geographically')
    

    
    return X_train_scaled, X_validate_scaled, X_test_scaled
    

    
    
    
    
