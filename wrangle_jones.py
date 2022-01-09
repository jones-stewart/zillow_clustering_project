##########################################################################################| WRANGLE.PY

##########################################################################################| IMPORTS
import os

from env import host, user, password

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##########################################################################################| ACQUIRE FUNCTION
def acquire_zillow():
    '''
THIS FUNCTION CHECKS FOR A LOCAL ZILLOW CSV FILE AND WRITES IT TO A DF. 
IF A LOCAL ZILLOW CSV FILE DOES NOT EXIST IT RUNS THE QUERY TO ACQUIRE
THE DATA USING THE SQL SERVERS CREDENTIALS FROM THE ENV FILE THROUGH
THE CONNECTION STRING.
    '''
    
    if os.path.isfile('zillow.csv'):
        zillow = pd.read_csv('zillow.csv', index_col = 0)
        return zillow
    
    else: 
        db = 'zillow'
        query  = '''
            SELECT bathroomcnt as baths, bedroomcnt as beds, calculatedfinishedsquarefeet as sqft, fullbathcnt as fullbaths, latitude,
                   longitude, yearbuilt, taxvaluedollarcnt as tax_value, logerror, transactiondate, 
                   unitcnt, propertylandusetypeid
            FROM properties_2017
            LEFT JOIN predictions_2017 pred USING(parcelid)
            LEFT JOIN airconditioningtype USING(airconditioningtypeid)
            LEFT JOIN architecturalstyletype USING(architecturalstyletypeid)
            LEFT JOIN buildingclasstype USING(buildingclasstypeid)
            LEFT JOIN heatingorsystemtype USING(heatingorsystemtypeid)
            LEFT JOIN propertylandusetype USING(propertylandusetypeid)
            LEFT JOIN storytype USING(storytypeid)
            LEFT JOIN typeconstructiontype USING(typeconstructiontypeid)
            WHERE latitude IS NOT NULL
            AND longitude IS NOT NULL
            AND transactiondate LIKE "2017%%"
            AND pred.id IN (SELECT MAX(id)
            FROM predictions_2017
            GROUP BY parcelid
            HAVING MAX(transactiondate));
            '''
        url = f'mysql+pymysql://{user}:{password}@{host}/{db}'
        zillow = pd.read_sql(query, url)
        zillow.to_csv('zillow.csv')
        return zillow

##########################################################################################| CLEAN FUNCTIONS
def wrangle_zillow(df):
    '''
THIS FUNCTION TAKES IN A RAW DATAFRAME AND PERFORMS THE FOLLOWING DATA CLEANING:
    1) FILTER OUT NON-SINGLE UNIT OBSERVATIONS
    2) DROPS NULLS BASED ON THRESHOLDS
    3) CREATE AGE COLUMN
    4) DROP REMAINING ROWS WITH NULL VALUES
    5) CORRECTING DTYPES
    6) DROP COLUMNS
    7) REMOVE OUTLIERS
    '''
    
    
    #1) filter single units by properylandusetype, bath, bed, and sqft count, and unit count
    df = df[df.propertylandusetypeid.isin([261, 262, 263, 264, 266, 268, 273, 276, 279])]
    df = df[(df.baths > 0) & (df.beds > 0) & (df.sqft > 300)]
    
    #2) dropping null rows and columns with > 50% of values missing
    df = df.dropna(axis = 1, thresh = .5 * len(df))
    df = df.dropna(thresh = .5 * len(df.columns))
    
    #3) create age column from yearbuilt
    df['age'] = 2022 - df.yearbuilt
    
    #4) drop any remaining rows with null values
    df.dropna(inplace = True)
    
    #5) correcting dtypes
    df[['beds', 'sqft', 'fullbaths', 'latitude', 'longitude', 'yearbuilt', 'unitcnt']] = df[['beds', 'sqft',\
                                                                                                    'fullbaths', 'latitude', 'longitude', 'yearbuilt', 'unitcnt']].astype('int')
    
    #6) drop columns 
    df = df.drop(columns = ['propertylandusetypeid', 'transactiondate', 'yearbuilt', 'unitcnt'])
    
    #7) remove outliers
    for col in df[['baths', 'beds', 'sqft', 'fullbaths', 'tax_value', 'logerror']]:
        
        if df[col].dtype != 'O':
            # get quartiles
            q1, q3 = df[col].quantile([.25, .75])

            # compute iqr
            iqr = q3 - q1

            # get cutoff points for removing outliers
            upper = q3 + 1.5 * iqr
            lower = q1 - 1.5 * iqr

            # remove outliers
            df = df[(df[col]>lower)&(df[col]<upper)]
    
    
    return df
    

##########################################################################################| PREPARE FUNCTION
def split_data(df):
    '''
THIS FUNCTION TAKES IN A CLEAN DF AND SPLITS IT, RETURNING TRAIN, VALIDATE, AND TEST DFs
    '''
    
    train, test = train_test_split(df, train_size = 0.8, random_state = 123)
    train, validate = train_test_split(train, train_size = 0.75, random_state = 123)
    
    return train, validate, test

def scale_data(train, validate, test, df):
    '''
THIS FUNCTION TAKES IN A TRAIN, VALIDATE, AND TEST DF, ALONG WITH THE ORIGINAL, UNSPLIT DF AND RETURNS 
THE FOLLOWING 6 DFS:
    TRAIN, VALIDATE AND TEST: USING THE SPLIT_DATA() FUNCTION
    EACH OF THE ABOVE THREE, BUT ADDED COLUMNS WHERE THE DATA IS SCALED
THIS FUNCTION SCALED THE DATA USING THE STANDARD SCALER WHICH NORMALLY DISTRIBUTES EACH COLUMN'S VALUES
AROUND 0

IF NEEDED, WILL COME BACK LATER AND ADD CODE THAT DROPS THE ORIGINAL, UNSCALED COLUMNS FROM THE SCALED
DFS.
    '''
    
    train_scaled = train
    validate_scaled = validate
    test_scaled = test
    
    cols_to_scale = ['baths', 'beds', 'sqft', 'fullbaths', 'tax_value', 'logerror', 'age']
    
    scaler = StandardScaler()
    scaler.fit(train_scaled[cols_to_scale])
    
    train_scaled[[f'{col}_scaled' for col in cols_to_scale]] = scaler.transform(train_scaled[cols_to_scale])
    validate_scaled[[f'{col}_scaled' for col in cols_to_scale]] = scaler.transform(validate_scaled[cols_to_scale])
    test_scaled[[f'{col}_scaled' for col in cols_to_scale]] = scaler.transform(test_scaled[cols_to_scale])
    
    train, validate, test = split_data(df)
    
    return train, validate, test, train_scaled, validate_scaled, test_scaled

    
    

