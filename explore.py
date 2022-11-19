# import modules
import pandas as pd
import seaborn as sns

import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from scipy.stats import f_oneway
############################################### optional features#################################
def loliplot(train):
    # create data frame for loliplot
    loli= pd.DataFrame(
        {'home_feature':['No_Feature','Fire place','Garage','Any_Feature','Pool','Deck'],
         'avg_home_value':[train[train.optional_features==0].home_value.mean(),train[train.fireplace==1].home_value.mean(),train[train.garage==1].home_value.mean(),
                          train[train.optional_features==1].home_value.mean(), train[train.pool==1].home_value.mean(),
                          train[train.deck==1].home_value.mean()]
        })
    # set fig size
    fig, axes = plt.subplots(figsize=(7,6))
    # set font and style
    sns.set(font_scale= 5.5) 
    sns.set_theme('talk')
    sns.set_style('white')

    # using subplots() to draw vertical lines
    axes.vlines(loli['home_feature'], ymin=0, ymax=loli['avg_home_value'],color='olive',lw=4)


    # drawing the markers (circle)
    axes.plot(loli['home_feature'], loli['avg_home_value'], "D",color ='plum',markersize=13) 
    axes.set_ylim(0)

    # formatting axis and details 
    plt.xlabel('')
    plt.ylabel('Average Home Value', fontsize =20)
    plt.title('Loli plot',fontsize =35)
    plt.xticks(loli['home_feature'])
    axes.set_yticks(ticks=[0,175_000, 350_000,525_000,700_000])
    
    return plt.show();
########################################## loli Median
def lolipop_plot(train):# no features #any features , garage ,fireplace, pool, deck
    # create data frame for loliplot
    loli= pd.DataFrame(
        {'home_feature':['No_Feature','Any_Feature','Garage','Fireplace','Pool','Deck'],
         'avg_home_value':[train[train.optional_features==0].home_value.median(),
                            train[train.optional_features==1].home_value.median(),
                            train[train.garage==1].home_value.median(),
                          train[train.fireplace==1].home_value.median(),
                           train[train.pool==1].home_value.median(),
                          train[train.deck==1].home_value.median()]
        })
    # set fig size
    fig, axes = plt.subplots(figsize=(10,5))
    # set font and style
    #sns.set(font_scale= 5.5) 
    sns.set_theme('talk')
    sns.set_style('white')

    # using subplots() to draw vertical lines
    axes.vlines(loli['home_feature'], ymin=0, ymax=loli['avg_home_value'],color='olive',lw=4)


    # drawing the markers (circle)
    axes.plot(loli['home_feature'], loli['avg_home_value'], "D",color ='plum',markersize=13) 
    axes.set_ylim(0)

    # formatting axis and details 
    plt.xlabel('')
    plt.ylabel('Median Home Value', fontsize =20)
    plt.title('Loli plot',fontsize =35)
    plt.xticks(loli['home_feature'],fontsize = 15)
    plt.yticks(fontsize = 15 )
    axes.set_yticks(ticks=[0,175_000, 350_000,525_000,700_000])
    
    return plt.show();

#################################################### more house ##################################
def more_house (train):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle('More House equals More Home Value')

    g = sns.regplot(ax=axes[0],x='bedrooms', y='home_value', data=train,color='olive',
                scatter_kws={'s': 100, 'alpha': 0.5, 'color': 'plum'}
                )
    h = sns.regplot(ax=axes[1],x='bathrooms', y='home_value', data=train,color='olive',
                scatter_kws={'s': 100, 'alpha': 0.5, 'color': 'plum'}
                )
    i = sns.regplot(ax=axes[2],x='squarefeet', y='home_value', data=train, color='olive',
                scatter_kws={'s': 100, 'alpha': 0.5, 'color': 'plum'}
                )
    g.yaxis.set_major_formatter(ticker.EngFormatter())
    g.set( ylabel = 'Home Value', xlabel='Bedrooms')
    h.set( ylabel = '',xlabel ='Bathrooms')
    i.set( ylabel = '', xlabel ='Square Feet')

    plt.yticks(ticks=[0,500_000, 1_000_000,1_500_000,2_000_000]);

########################################## County ##########################

def county_vs_homevalue(df):
    ''' This function takes in zillow data frame and returns a boxplot that
        shows the difference in home value by county.
    '''
    df['county'] = df.county.replace({6037:'Los Angeles',
                       6059:'Orange',
                       6111:'Ventura'})
    
    # create boxplot and set parameters
    g=sns.boxplot(x='home_value', y='county',data=df,color='plum',palette='PiYG')
    # set backgorund and text scale
    sns.set(font_scale=1.3)  
    g.set_ylabel(ylabel='')
    g.set_xlabel(xlabel='Home Value')
    sns.set_theme('talk')
    sns.set_style('white')
    g.xaxis.set_major_formatter(ticker.EngFormatter())
   
    plt.xticks(ticks=[0,500_000, 1_000_000,1_500_000,2_000_000])
    # create title 
    plt.title('The difference in home value',fontsize=25,fontweight=100)
  
    plt.show(g)



def anova_county_test(train):
    f,pval = f_oneway(train[train['county']=='Los Angeles'].home_value,
                                    train[train['county']=='Orange'].home_value,
                                    train[train['county']=='Ventura'].home_value)

    print(f't-stat {f}')
    print(f'p-value {pval}')
############################################# home age ################################

def home_scatterplot(train):
    # get home_age data frame
    home_df = setup_homeage(train)
    
    # set fig size
    fig, axes = plt.subplots(figsize=(7,6))

    # scatter plot
    g= sns.scatterplot(data = home_df, x='Age',y='Value', color='olive',s=300, marker='1',linewidth=1.5)

    # set backgorund and text scale
    sns.set(font_scale=1.3)  
    g.set_xlabel(xlabel='Home Age')
    g.set_ylabel(ylabel='Home Value')
    sns.set_theme('talk')
    sns.set_style('white')
    g.yaxis.set_major_formatter(ticker.EngFormatter())
    plt.yticks(ticks=[0,500_000, 1_000_000,1_500_000,2_000_000])
    # create title 
    plt.title('Older Home less Home Value',fontsize=25,fontweight=100);

def setup_homeage(train):
    # List1
    age =range(2,140)

    # List2
    value = []
    for i in age:
        value.append(train[train.home_age == i].home_value.median())

    # get the list of tuples from two lists.
    # and merge them by using zip().
    list_of_tuples = list(zip(age, value))

    # Assign data to tuples.
    list_of_tuples

    # pandas Dataframe.
    home_df = pd.DataFrame(list_of_tuples,
                      columns=['Age', 'Value'])
    return (home_df)