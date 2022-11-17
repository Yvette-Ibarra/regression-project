import env
import os
import numpy as np
import pandas as pd

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
   
         # sql query for acquisition
        sql_query = """
        SELECT calculatedfinishedsquarefeet,bathroomcnt,bedroomcnt,taxvaluedollarcnt,yearbuilt, fireplacecnt,
        decktypeid, poolcnt, garagecarcnt,fips

        FROM properties_2017
        LEFT JOIN propertylandusetype USING(propertylandusetypeid)
        LEFT JOIN predictions_2017 USING(parcelid)
        WHERE (propertylandusetype.propertylandusedesc LIKE ('%%Single%%')) 
            AND (predictions_2017.transactiondate like '2017%%');
        """

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
################################################## Prepare Data #######################
def zillow_prep(df):
    
    # remove outliers
    df = handle_outliers(df)
    
    # removed rows with 0 beds and 0 baths
    df = df[~(df.bathrooms==0) & ~(df.bedrooms ==0)]
    
    # process nulls in luxury features:
    df = process_fancy_features(df)
    
    # drop nulls
    df = df.dropna()

    return df



###################################### Outliers ##########################
def handle_outliers(df):
    """Manually handle outliers '"""
    df = df[df.bathrooms <= 6]
    
    df = df[df.bedrooms <= 6]
    
    df = df[df.home_value <= 1_750_000]
    
    return df



####################################### Features ###########################
def process_fancy_features(df):
    
    columns = ['fireplace','deck','pool','garage']    
    for feature in columns:
        df[feature]=df[feature].replace(r"^\s*$", np.nan, regex=True)     
        # fill fancy features with 0 assumption that if it was not mark it did not exist
        df[feature] = df[feature].fillna(0)
    return df