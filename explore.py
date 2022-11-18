# import modules
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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