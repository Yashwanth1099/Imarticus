import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import capstode_required_func as cf


def stack_bar(df,columns ,col2 = 'TARGET'):
    
    '''
    df = dataframe,
    col1 = list - columns to be plotted,
    col2 = hue 
    
    '''
    
    
    for col1 in columns:
        
        if df[col1].dtypes == 'int64' :
            df[col1] = df[col1].astype('O')
            
        print(df[col1].dtypes,col1)
        
        fig, ax = plt.subplots(figsize=(8,8))
        
        cross = pd.crosstab(df[col1],df[col2])
         # now stack and reset
        stacked = cross.stack().reset_index().rename(columns={0:'value'})
         # plot grouped bar chart
        sns.barplot(x=stacked[col2] , y=stacked['value'], hue=stacked[col1])
        
        plt.show()
        
        
        
def missing_per_class(df,columns ,col2 = 'TARGET'):
    
    '''
    df = dataframe,
    col1 = list - columns to be plotted,
    col2 = hue 
    
    '''
    
    for col1 in columns:
            
        # print(df[col1].dtypes,col1)
        
        fig, ax = plt.subplots(figsize=(10,10))
        
        cross = pd.crosstab(df[col1].isnull(),df[col2])

         # now stack and reset
        stacked = cross.stack().reset_index().rename(columns={0:'value'})
        
        # print(f" \n Split of missing value in {col1} ")
        # print(stacked[stacked[col1]==True][[col2,'value']].reset_index(),'\n')
        
         # plot grouped bar chart
        sns.barplot(x=stacked[col2] , y=stacked['value'], hue=stacked[col1])
        plt.show()
            


    

def dist_bar (df, columns ,col2 = "TARGET", hist = True, kde = True):
    
    '''
    plots the distribution of the columns with hue as col2
    
    parameters:
        
        df = dataframe,
        columns = list , (columns name to be plotted from df ),
        col2 = hue,
        hist = bool, True (deafault)  - plots Histogram
        kde =  bool, True (deafault) - plots KDE
    
    '''
    #plt.figure(figsize = (12, 6))

    for col1 in columns:
        fig, ax = plt.subplots(figsize=(8,6))
        # plt.subplot(1,2,1)
        plt.title('Distribution of data')
        sns.kdeplot(data=df, x=col1, hue=col2)

        # plt.subplot(1,2,2)
        # plt.title(' count vs spread of data')
        # sns.ecdfplot(data=df, x=col1, hue=col2, stat="count")
    
        plt.show ()    
    
    
    
def plot_outliers(df):
    
    '''
    
    plots the columns with outliers in the given df
    
    parametrs :
        
        df = dataframe
        
    returns : none
    
    
    '''
    df_c = df.copy()

    for col in df_c.columns:
        
        b_val = list(df_c[col].value_counts().index)     
        
        if ([0,1] == b_val )or['No', 'Yes'] == b_val or ([1,0] == b_val ) :

            df_c[col] = df_c[col].astype('bool')    
            
    
    outliers = cf.outlier_analysis(df_c)
    
    outlier_cols = []
    
    for k,v in outliers.items():
        
        if len(v)>0:
            
            outlier_cols.append(k)
            
    num_cols = df_c[outlier_cols].select_dtypes(include = ['float64']) #,'int64'
    
    for col in num_cols:
        
        
        fig, ax = plt.subplots(figsize=(18,9))

        plt.subplot(1,2,1)
        # plt.title('scatter plot')
        # sns.scatterplot(np.array(range(len(df_c))),df_c[col])

        # plt.subplot(1,2,2)
        
        plt.title('box plot')
        sns.boxplot(y=col,x='TARGET', data = df)
        
        plt.show()



def miss_viz(df , kind = 'all'):
    
    '''
    plots the missing values for missing value analysis, 
    shows the missing count in bar graph (bar), missing value corr 
    and pattern of missing value
    for all columns

    
    parameters:
        
        df = dataframe
        
        kind = 'all' (deafault) ,'bar' ,'matrix', 'dend' , 'corr'
        
    returns :  missing columns
    
    '''
    
    desc_df = cf.primary_analysis(df)
    missing_df = desc_df[desc_df['missing_cnt_rto']>0]
    missing_df = missing_df.set_index('name').sort_values(by =['missing_cnt_rto'], ascending = False)
    missing_columns = missing_df.index.values
    
    if kind == 'bar' or kind == 'all':
        msno.bar(df[missing_columns],log=True)
        plt.show()
        
    if kind == 'matrix' or kind == 'all':
        msno.matrix(df[missing_columns])
        plt.show()

        
    if kind == 'dend' or kind == 'all':
        msno.dendrogram(df[missing_columns])
        plt.show()

    
    if kind == 'corr' or kind == 'all':
        msno.heatmap(df[missing_columns])
        plt.show()
        
        