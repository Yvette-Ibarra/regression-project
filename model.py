import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


def model_data_prep(train, validate,test):
    '''model_data_prep takes in train validate,test and scales using scale_data and sets up
    features and target ready for modeling
    '''
    train = train.drop(columns=['yearbuilt','fireplace', 'deck', 'pool', 'garage'])
    validate = validate.drop(columns=['yearbuilt','fireplace', 'deck', 'pool', 'garage'])
    test = test.drop(columns=['yearbuilt','fireplace', 'deck', 'pool', 'garage'])
    
    X_train_scaled, X_validate_scaled, X_test_scaled = scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['squarefeet','bathrooms','bedrooms','home_age'])
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
               columns_to_scale=['squarefeet','bathrooms','bedrooms','home_age']):
    '''
    scale_data takes in train , validate, test data  and returns their scaled counterparts using MinMaxscaler.
    returns train_scaled, validate_scaled, test_scaled
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
############################################### MODELS ######################################

def poly_d2(X_train,y_train): 
    ''' Model poly_d2 is a Polynomial Regressor with degree 2 takes in x_train and y_train and 
     fits into train data, transforms X_train
     returns lm2, X_train_poly2, y_train'''
    # Generate polynomial features  
    poly2 = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly2.fit(X_train)
    X_train_poly2 = pd.DataFrame(
        poly2.transform(X_train),
        columns=poly2.get_feature_names(X_train.columns),
        index=X_train.index, )
 
    # 2. Use the features
    lm2 = LinearRegression()
    lm2.fit(X_train_poly2, y_train)

    return lm2, X_train_poly2,poly2

def poly_d2i(X_train, y_train):  
    ''' Model poly_d2i is a Polynomial Regressor with degree 2 wit interactions only takes in x_train and y_train and 
     fits into train data, transforms X_train
     returns lm2, X_train_poly2, y_train
    '''  
    # 1. Generate Polynomial Features
    poly2i = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly2i.fit(X_train)
    X_train_poly2i = pd.DataFrame(
        poly2i.transform(X_train),
        columns=poly2i.get_feature_names(X_train.columns),
        index=X_train.index,)

    # 2. Use the features
    lm2i = LinearRegression()
    lm2i.fit(X_train_poly2i, y_train)

    return lm2i,X_train_poly2i,poly2i


def poly_d3(X_train, y_train):  
    ''' Model poly_d3 is a Polynomial Regressor with degree 3 takes in x_train and y_train and 
    fits into train data, transforms X_train
    returns lm2, X_train_poly2, y_train
    '''    
    # 1. Generate Polynomial Features
    poly3 = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
    poly3.fit(X_train)
    X_train_poly3 = pd.DataFrame(
        poly3.transform(X_train),
        columns=poly3.get_feature_names(X_train.columns),
        index=X_train.index, )

    # 2. Use the features
    lm3 = LinearRegression()
    lm3.fit(X_train_poly3, y_train)

    return lm3,X_train_poly3,poly3

def poly_d4(X_train, y_train): 
    ''' Model poly_d4 is a Polynomial Regressor with degree 4 takes in x_train and y_train and 
    fits into train data, transforms X_train
    returns lm2, X_train_poly2, y_train
    '''   
    # 1. Generate Polynomial Features
    poly4 = PolynomialFeatures(degree=4, include_bias=False, interaction_only=False)
    poly4.fit(X_train)
    X_train_poly4 = pd.DataFrame(
        poly4.transform(X_train),
        columns=poly4.get_feature_names(X_train.columns),
        index=X_train.index,)
  
    # 2. Use the features
    lm4 = LinearRegression()
    lm4.fit(X_train_poly4, y_train)

    return lm4,X_train_poly4,poly4

##################################### predictions data frame ##########################

def predictions(X_train,y_train,X_validate,y_validate,X_test, y_test):
    ''' predictions takes in X_train,y_train,X_validate,y_validate,X_test, y_test)
    creates a data frame with predictions using models
    returns train_pred, validate_pred, test_pred'''
    # get fitted models
    lm2, X_train_poly2, poly2 = poly_d2(X_train,y_train)
    lm3, X_train_poly3,poly3 = poly_d3(X_train, y_train)
    lm2i, X_train_poly2i,poly2i = poly_d2i(X_train, y_train)
    lm4, X_train_poly4,poly4 = poly_d4(X_train, y_train)

    # set up dataframe for predictions, add actual values
    train_pred = pd.DataFrame({
        'actual': y_train
    }) 
    validate_pred = pd.DataFrame({
        'actual': y_validate
    }) 

    test_pred = pd.DataFrame({
        'actual': y_test
    }) 

    # add a baseline model
    #train_pred['baseline_mean'] = y_train.mean()
    #validate_pred['baseline_mean'] = y_train.mean()
    #test_pred['baseline_mean'] = y_train.mean()
    
    #add baselin median
    train_pred['baseline_median'] = y_train.median()
    validate_pred['baseline_median'] = y_train.median()
    test_pred['baseline_median'] = y_train.median()


    # Add degree 2 to data frame
    train_pred['poly_d2'] = lm2.predict(X_train_poly2)
    X_validate_poly2 = poly2.transform(X_validate)
    validate_pred['poly_d2'] = lm2.predict(X_validate_poly2)
  
     
    # Add degree 2 interactions only to data frame
    train_pred['Ipoly_d2'] = lm2i.predict(X_train_poly2i)
    X_validate_poly2i = poly2i.transform(X_validate)
    validate_pred['Ipoly_d2'] = lm2i.predict(X_validate_poly2i)
   

    # Add degree 3  to data frame
    train_pred['poly_d3'] = lm3.predict(X_train_poly3)
    X_validate_poly3 = poly3.transform(X_validate)
    validate_pred['poly_d3'] = lm3.predict(X_validate_poly3)

    # add test to degree 3
    X_test_poly3 = poly3.transform(X_test)
    test_pred['poly_d3'] = lm3.predict(X_test_poly3)

    # Add degree 4  to data frame
    train_pred['poly_d4'] = lm4.predict(X_train_poly4)
    X_validate_poly4 = poly4.transform(X_validate)
    validate_pred['poly_d4'] = lm4.predict(X_validate_poly4)
  
    
    return train_pred, validate_pred, test_pred

######################################### evaluation metrics ####################################
def evaluate_metrics(df, col,actual):
    ''' evalate_metrics takes in a dataframe columns and actual(target)
    calculates MSE, SSE, RMSE, ESS, TSS, R2 and 
    returns  MSE, SSE, RMSE,ESS, TSS,R2'''
    MSE = mean_squared_error(actual, df[col])
    SSE = MSE * len(df)
    RMSE = MSE ** .5
    ESS = ((df[col] - actual.mean())**2).sum()
    TSS = ESS + SSE
    R2 = explained_variance_score(actual, df[col])
    return MSE, SSE, RMSE,ESS, TSS,R2

def metric_train(train_pred, y_train): 
    ''' metric_train takes in train_pred, y_train and uses evaluate_metrics
    to calculate RMSE and R2 for each column using evaluate_metrics and creates
    dataframe with calculations
    returns metric_train
    '''
    # create columns
    col = train_pred.columns.to_list()
    metric_train = pd.DataFrame(columns =['model','train_RMSE','train_R2'])
    for i in col:
        MSE,SSE, RMSE, ESS, TSS, R2 = evaluate_metrics(train_pred, i , y_train)
        # sklearn.metrics.explained_variance_score
        RMSE = RMSE.round()
        metric_train= metric_train.append({
                        'model': i,
                         'train_RMSE':RMSE,
                         'train_R2':R2},ignore_index=True)
        
    return metric_train

def metric_validate(validate_pred, y_validate): 
    ''' metric_validate takes in validate_pred, y_validate and uses evaluate_metrics
    to calculate RMSE and R2 for each column using evaluate_metrics and creates
    dataframe with calculations
    returns metric_validate
    '''
    col = validate_pred.columns.to_list()
    metric_val = pd.DataFrame(columns =['model','val_RMSE','val_R2'])
    for i in col:
        MSE,SSE, RMSE, ESS, TSS, R2 = evaluate_metrics(validate_pred, i , y_validate)
        # sklearn.metrics.explained_variance_score
        RMSE = RMSE.round()
        metric_val= metric_val.append({
                        'model': i,
                         'val_RMSE':RMSE,
                         'val_R2':R2},ignore_index=True)
        
    return metric_val
    
def metrics(train_pred,y_train, validate_pred, y_validate):
    ''' metric takes in train_pred,y_train, validate_pred, y_validate  and concats
    dataframes of evaluations for final report of models
    returns metric
    '''
    # get models metrics on train
    train_metric = metric_train(train_pred,y_train)
    
    # get models metrics on validata data, sorted by R^2
    val_metric = metric_validate(validate_pred,y_validate)
    val_metric.drop(columns='model', inplace = True)
    
    # concatinate data frames
    metric = pd.concat([train_metric, val_metric], axis=1)
    return metric

def metric_test(test_pred, y_test): 
    ''' metric_test takes in test_pred, y_test and uses evaluate_metrics
    to calculate RMSE and R2 for each column using evaluate_metrics and creates
    dataframe with calculations
    returns metric_test
    '''
    col = test_pred.columns.to_list()
    metric_test = pd.DataFrame(columns =['model','test_RMSE','test_R2'])
    for i in col:
        MSE,SSE, RMSE, ESS, TSS, R2 = evaluate_metrics(test_pred, i , y_test)
        # sklearn.metrics.explained_variance_score
        RMSE = RMSE.round()
        metric_test= metric_test.append({
                        'model': i,
                         'test_RMSE':RMSE,
                         'test_R2':R2},ignore_index=True)
    return metric_test