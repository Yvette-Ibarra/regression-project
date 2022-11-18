import pandas as pd


from sklearn.preprocessing import MinMaxScaler


def model_data_prep(train, validate,test):
    X_train_scaled, X_validate_scaled, X_test_scaled = scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['squarefeet','bathrooms','bedrooms','yearbuilt','home_age'])
    # Setup X and y
    X_train_scaled = X_train_scaled.drop(columns=['home_value','county'])
    y_train = train.home_value

    X_validate_scaled = X_validate_scaled.drop(columns=['home_value','county'])
    y_validate = validate.home_value

    X_test_scaled = X_test_scaled.drop(columns=['home_value','county'])
    y_test = test.home_value
    
    return X_train_scaled,y_train, X_validate_scaled,y_validate, X_test_scaled, y_test



def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['squarefeet','bathrooms','bedrooms','yearbuilt','home_age']):
    '''
    scale_data takes in train , validate, test data  and returns their scaled counterparts.
    '''
    # create copies of our original data
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #create the scaler
    scaler = MinMaxScaler()
    # fit the scaler into train data
    scaler.fit(train[columns_to_scale])
    
    # applying the scaler to train, validate, and test data
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    return train_scaled, validate_scaled, test_scaled


