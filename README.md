# Zillow Home Value
# Project Description
Zillow Group  is an American tech real-estate marketplace company that focus on all the stages of the home lifecycle: renting, buying, selling, financing and home improvement. Zillow is dedicated to empowering consumers with data, inspiration and knowledge around the place they call home, and connecting them with the best local professionals who can help. We will be looking into Zillow data from 2017 of single family properties and make a model to predict the home values.

# Project Goal
* Discover key attributes that drive and have a high correlation with home value.

* Use those attributes to develop a machine learning model to predict home value.

    * Carefully select features that will prevent data leakage. 

    * Data leakage refers to a mistake that is made by the creator of a machine learning model in which information about the target variable is leaking into the input of the model during the training of the model; information that will not be available in the ongoing data that we would like to predict on.

# Initial Thoughts
My initial hypothesis is that bathrooms  and squarefeet are strong drivers of home value.

# The Plan
* Acquire data from Codeup database
* Prepare data
* Explore data in search of drivers of home_value
    * Answer the following initial question:
        * How do features of a home influence home value?
        * Does more house equal more home value?
        * Does county location make a difference in home value?
        * Is home age a driver of home value?

* Develop a Model to predict home value.
    * Use drivers identified in explore to build predictive models of home value and using Kbest and/or RFE 
    * Evaluate models on train and validate data using RMSE (Root mean square Error)
    * Select the best model based on the least RMSE
    * Evaluate the best model on test data
* Draw conclusions


# Data Dictionary

 Feature          | Description|
| :---------------: | :---------------------------------- |
| home_value | The total tax assessed value of the parcel  |
| squarefeet:  | Calculated total finished living area of the home |
| bathrooms:   |  Number of bathrooms in home including fractional bathrooms |
| bedrooms: | Number of bedrooms in home  |
| yearbuilt:  |  The Year the principal residence was built   |
| fireplace: | fireplace on property (if any) |
| deck:  | deck on property (if any) |
| pool:  | pool on property (if any) |
| garage: | garage on property (if any) |
| county: | FIPS code for californian counties: 6111 Ventura County, 6059  Orange County, 6037 Los Angeles County |
| home_age: | The age of the home in 2017 |
| optional_features: | If a home has any of the follwing: fireplace, deck, pool, garage it is noted as 1 |
| additional features: | Encoded and values for categorical data |

# Steps to Reproduce
1. Clone this repository
2. Get Zillow data from Codeup Database:
    * Must have access to Codeup Database
    * Save a copy env.py file containing Codeup: hostname, username and password
    * Save file in cloned repository
3.Run notebook

# Takeaways and Conclusions
* Homes with an optional feature such as deck, pools, garage, fireplace have more value.
* Home with more bedrooms and bathrooms tend to have more value on average.
* County location make a difference in home value.
* Home age has a relationship with home value.

# Recommendations
Standardize methods of data collection to increase data accuracy; this will likely improve models ability to predict future home values.

