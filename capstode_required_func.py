import pandas as pd
import numpy as np
import mlcp.pipeline as pl
import missingno as msno
import scipy.stats
import seaborn as sns
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def read_data(filepath):
    df = pd.read_csv(filepath)
    df_name = str(filepath).split('/')[-1]
        
    print(f' \n {df_name} contains {df.shape [0]} rows and {df.shape [1]} columns ')
    return df


def one_hot_enc(df, nan_as_category ):
    
    '''
    parameter:
    df = dataframe
    nan_as_category = boolean (true - considers nan as a category , false - ignores nan)
    
    returns: 
    dataframe with label encoded
    list of categories newly added
    
    '''
    
    df_dum = df
    cat_col = [col for col in df.select_dtypes(include='O')]
    num_col = [col for col in df.select_dtypes(exclude = 'O' )]
    df_dum = pd.get_dummies(df, columns = cat_col, dummy_na = nan_as_category)
    #new_cols = [col for col in df_dum.columns if col not in df.columns] 
    
    return df_dum#, new_cols


def primary_analysis(df, file_name= None):
    
  
    desc_df = {'name' : [],
              'unique' : [],
              'missing_cnt':[],
              'missing_cnt_rto':[],
              'type' :[],
              'skew':[],
              'outlier_count':[],
              }

    
    outliers = outlier_analysis(df)

    for c in df:
        
        #print(c)
        name = df[c]
        unique = df[c].nunique()
        missing_cnt = df[c].isnull().sum()
        missing_cnt_rto = (missing_cnt/len(df))*100
        type_col = df[c].dtypes


        
        #if(type_col == 'int32' or  type_col == 'float32'):
        if(type_col != 'O'):
            skew = df[c].skew()   
            for k,v in outliers.items():
                if k == c:
                    outlier_count = len(v)
         
        else:
            skew = np.nan
 
            outlier_count =np.nan
        

        desc_df['name'].append(c)
        desc_df['unique'].append(unique)
        desc_df['missing_cnt'].append(missing_cnt)
        desc_df['missing_cnt_rto'].append(missing_cnt_rto)
        desc_df['type'].append(type_col)
        desc_df['skew'].append(skew)
        desc_df['outlier_count'].append(outlier_count)

    desc_df  = pd.DataFrame(desc_df) 
    desc_df =  desc_df.sort_values(by='missing_cnt_rto', ascending = False)
    #print(file_name ,' head \n ',desc_df.head(5),'\n')

    return desc_df



def missing_value_analysis(df,drop_th):
  samples = len(df); 
  dropped_col=[]
  dfc=df.copy(); dfc=dfc.dropna()
  missing_dict = {}
  for c in df.columns:
      n_missing = df[c].isnull().sum()
      missing_ratio = n_missing/samples
#        x_scores=feature_scores(dfc.drop([c],axis=1), dfc[c]
      if missing_ratio >= drop_th:
          df = df.drop([c], axis=1)
          dropped_col.append(c)
      elif n_missing > 0:
          missing_dict[c] = []
          missing_dict[c].append((n_missing, missing_ratio, df[c].dtype))
          
  print(f'''Total number of dropped columns for threshold {drop_th}: {dfc.shape[1]-df.shape[1]}
Total number of columns with missing value :{len(missing_dict)}''')

  print('Columns dropped :',dropped_col)
          
  #print("dropped_columns-->", dropped_col,"\n")
  # print("\n other missing colums-->")
  # for k,v in missing_dict.items():
  #     print(k, " === " , v[0][1]); print("\n")
      
  return df




def convert_types(df, print_info = False):
    
    original_memory = df.memory_usage().sum()
    
    # Iterate through each column
    for c in df:
         
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
        
    return df




def aggregate(df, primary_id, df_name):
    
    """
    Groups and aggregates the numeric values in a child dataframe
    by the parent variable.
    
    Parameters
    --------
        df (dataframe): 
            the child dataframe to calculate the statistics on
        primary_id (string): 
            the parent variable used for grouping and aggregating
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated by the `primary_id` for 
            all columns. 
    
    """
    bool_col = []
    for col in df.columns:
        if ([0,1] == list(df[col].value_counts().index) ):
            df[col] = df[col].astype('bool')
            bool_col.append(col)
    
    # Remove id variables other than grouping variable
    for col in df:
        if col != primary_id and 'SK_ID' in col:
            df = df.drop(columns = col)
            
            
    # Group by the specified variable and calculate the statistics for all values based on types
    
    numeric_df = df.select_dtypes(exclude = 'object').copy()
    
    cat_df = [col for col in df.select_dtypes(include='O')]
    cat_col = [col for col in cat_df if col not in bool_col ]
    df_enc = pd.get_dummies(df, columns = cat_col, dummy_na = False)
    cat_new_col = [col for col in df_enc.columns if col not in df.columns] 


    def mode(x):
        return lambda x: x.value_counts().index[0]
          
    agg_dict = {}
    
    for col in numeric_df.columns:
        agg_dict[col] = ['min','max','mean']
    
    for col in bool_col:
        agg_dict[col] = [ 'mean']
    
    for col in cat_new_col:
        agg_dict[col] = ['mean']
        

    agg = df_enc.groupby(primary_id).agg(agg_dict)
        
    agg[df_name+'count'] = df_enc.iloc[:,:2].groupby(primary_id).count().iloc[:,:1]

    agg.columns = pd.Index([ df_name+'_' +str(txt[0])+'_'+ str(txt[1]).upper() for txt in agg.columns])
    agg = agg.reset_index()
    
    return agg
        

def combine_df(df_1, df_2, primary_id, df_2_name) :
      
    '''  
    
    Parameters
    ----------
    
    df_1: main dataframe 
    df_2: dataframe to be joined
    primary_id : used to join the dataframe
    df_2_name = name for the dataframe
    
    returns
    -------
    
    mearged dataframe 
    
    prints the shape of new df
    
    '''
    #df_2 , df2_cols= one_hot_enc(df_2, nan_as_category=True)
    agg = aggregate(df_2,primary_id=primary_id,df_name=df_2_name)
    main_df = pd.merge(df_1,agg,how='left', on=primary_id)
    
    print(f' shape of merged df = {main_df.shape}')
    
    return main_df





def outlier_analysis(df):
    
    outlier_dict = {}
    
    for col in df.select_dtypes(exclude = 'O').columns: 
        Q1 = np.percentile(df[col], 25, interpolation = 'midpoint')
        Q3 = np.percentile(df[col], 75, interpolation = 'midpoint')
        
        IQR = Q3 - Q1
                
        low_lim = Q1 - 1.5 * IQR
        up_lim = Q3 + 1.5 * IQR
        
        outlier_dict[col] = []
        for x in df[col]:
            if ((x> up_lim) or (x<low_lim)):
                 outlier_dict[col].append(x)
        # print(' outlier in the dataset is', outlier_dict[col])
    return outlier_dict




