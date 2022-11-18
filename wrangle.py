import env
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

########################################## Acquire ##########################################
def fresh_zillow_data():
    '''
    This reads the zillow 2017 properties data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query = """
    SELECT 
    taxvaluedollarcnt AS home_value,
    calculatedfinishedsquarefeet AS squarefeet,
    bathroomcnt AS bathrooms,
    bedroomcnt AS bedrooms,
    yearbuilt, 
    fireplacecnt AS fireplace,
    decktypeid AS deck, 
    poolcnt AS pool, 
    garagecarcnt AS garage,
    fips AS county

    FROM properties_2017
    LEFT JOIN propertylandusetype USING(propertylandusetypeid)
    LEFT JOIN predictions_2017 USING(parcelid)
    WHERE (propertylandusetype.propertylandusedesc LIKE ('%%Single%%')) 
        AND (predictions_2017.transactiondate like '2017%%');
    """


    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, env.get_connection(db = 'zillow'))

    return df

def get_zillow_data(new = False):
        ''' Acquire Zillow data using properties_2017 table from Code up Data Base. Columns bedroomcnt, 
            bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips 
        '''
   

        filename = 'zillow.csv'

        # obtain cvs file
        if (os.path.isfile(filename) == False) or (new == True):
            df = fresh_zillow_data()
            #save as csv
            df.to_csv(filename,index=False)

        #cached data
        else:
            df = pd.read_csv(filename)

        return df
################################################## Prepare Data ##################################
def zillow_prep(df):
    
    # remove outliers
    df = handle_outliers(df)
    
    # removed rows with 0 beds and 0 baths
    df = df[~(df.bathrooms==0) & ~(df.bedrooms ==0)]
    
    # process nulls in luxury features:
    df = process_optional_features(df)
    
    # drop nulls
    df = df.dropna()

    # feature engineer: Home_age and optiona_feature
    df = new_features(df)

    #encode categorical features or turn to 0 & 1:
    df = encode_features(df)

    # rename dummy county to matching county name
    df = rename_county(df)

    return df



###################################### Outliers 
def handle_outliers(df):
    """Manually handle outliers '"""
    df = df[df.bathrooms <= 6]
    
    df = df[df.bedrooms <= 6]
    
    df = df[df.home_value <= 1_750_000]
    
    return df



####################################### Features 
def process_optional_features(df):
    
    columns = ['fireplace','deck','pool','garage']    
    for feature in columns:
        df[feature]=df[feature].replace(r"^\s*$", np.nan, regex=True)     
        # fill optional features with 0 assumption that if it was not mark it did not exist
        df[feature] = df[feature].fillna(0)
    return df

def new_features(df):
    #Creating new column for home age using year_built, casting as float
    df['home_age'] = 2017- df['yearbuilt']
    df["home_age"] = df["home_age"].astype('float')
    
    df['optional_features'] = (df.garage==1)|(df.deck == 1)|(df.pool == 1)|(df.fireplace == 1)
    
    return df
    
def encode_features(df):
    df.fireplace = df.fireplace.replace({2:1, 3:1, 4:1, 5:1})
    df.deck= df.deck.replace({66:1})
    df.garage = df.garage.replace({2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 13:1,14:1})
    df.optional_features = df.optional_features.replace({False:0, True: 1})
    temp = pd.get_dummies(df['county'], drop_first=False)
    df = pd.concat([df, temp],axis =1)
    return df 

def rename_county(df):
    # 6111 Ventura County, 6059  Orange County, 6037 Los Angeles County 
    df = df.rename(columns={6111.0: 'ventura_county',6059.0: 'orange_county',
            6037: 'los_angeles_county'}) 
    return df
###################################### Split Data

def split_data(df):
    '''
    split_data takes in data Frame and splits into  train , validate, test.
    The split is 20% test 80% train/validate. Then 30% of 80% validate and 70% of 80% train.
    Aproximately (train 56%, validate 24%, test 20%)
    Returns train, validate, and test 
    '''
    # split test data from train/validate
    train_and_validate, test = train_test_split(df, random_state=123, test_size=.2)

    # split train from validate
    train, validate = train_test_split(train_and_validate, random_state=123, test_size=.3)
                                   
    return train, validate, test