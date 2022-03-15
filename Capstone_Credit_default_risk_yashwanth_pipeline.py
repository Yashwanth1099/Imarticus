import mlcp.pipeline as pl
import mlcp.classifires as cl
import visual_plots as vp
import meta_data as md
import capstode_required_func as cf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
import seaborn as sns
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import re
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

#execution controls
classification=1; #if regression = 0, classification = 1
read=1
primary_analysis=1 #dev only
analyze_missing_values=0
treat_missing_values=1
observed_corrections=1
feature_engineering=1
visual_analysis=0
merge_data =1
define_variables=1
skew_corrections=1
scaling_imputing = 1
encoding=1
reduce_dim=1
oversample=1; #dev only
compare=0; #dev only
cross_validate=0; #dev only
train_classification=1

   
if read == 1:
    
  
    #main_df = meta.train_df()
    main_df = cf.read_data('Data/application_train.csv')
    y_name = 'TARGET'
    dtype_file = 'final_credit_anaysis.txt'
    df = main_df.copy()
       
    
    if classification == 1:
        
        sample_diff, min_y, max_y = pl.bias_analysis(df,y_name)
        print("sample diff:", sample_diff)
        print("sample ratio:", min_y/max_y)
        print(df[y_name].value_counts())
        
        plt.figure(figsize=(15,5))

        #plot 1:
        plt.subplot(1,2,1)
        plt.hist(df['TARGET'])
        plt.title('Spread of Target Data')
        plt.xlabel('Defaulters')
        
        #plot 2:
        
        plt.subplot(1,2,2)
        plt.pie(df['TARGET'].value_counts().values, labels= df['TARGET'].value_counts().index)
        plt.title('Spread of Target Data')
        plt.legend(['non defaulters','defaulters'])
        plt.show()
 
    

if primary_analysis==1:
    #consider: unwanted features, numerical conversions (year to no. years), 
    #wrong dtypes, missing values, categorical to ordinal numbers
    
    df_desc = cf.primary_analysis(df)
    df_outlier = cf.outlier_analysis(df)
    
    print('Number of duplicate entries : ',df.duplicated().sum())
 


if analyze_missing_values==1:
    
    #Analysing of missing values using graphs 
    
    print('\n \n \t \t Visualisation of missing columns ')
    missing_cols = df_desc[df_desc['missing_cnt']>0]['name'].values
    livarea = [ col for col in df.columns if 'AVG' in col or 'MEDI' in col or 'MODE' in col]
    other_cols = set(missing_cols) - set(livarea)
    miss_unique = np.where(df_desc[df_desc['missing_cnt_rto']>0].duplicated(subset=['missing_cnt_rto'])==True)
    
    
    print('\n \n \t \t Visualisation of missing columns by count ')
    plt.title('missing  values by features')
    vp.miss_viz(df[missing_cols],kind ='bar')

    print('\n \n \t \t Visualisation of missing columns by pattern ')
    vp.miss_viz(df[other_cols],kind ='matrix')
    
    print('\n \n \t \t Visualisation of missing columns by association ')
    vp.miss_viz(df[other_cols],kind ='dend')
    
    print('\n \n \t \t Visualisation of missing columns by correlation ')
    vp.miss_viz(df[other_cols],kind ='corr')
    
    
    print('\n \n \t \t Visualisation of missing values proportion with respect to Target variable ')
    vp.missing_per_class(df,df_desc.iloc[miss_unique]['name'])    
    
    #missing not at random
    print('\n \n value_count for FLAG_OWN_CAR\n',df['FLAG_OWN_CAR'].value_counts())
    print('\n\n total missing values in OWN_CAR_AGE = ',df['OWN_CAR_AGE'].isnull().sum())
    
    
    cross_tab = pd.crosstab(df['OCCUPATION_TYPE'],df['NAME_EDUCATION_TYPE'])
    
    cross_tab_T = pd.crosstab(df['TARGET'],df['NAME_EDUCATION_TYPE'])
    
    df[df['TARGET']==1]['AMT_REQ_CREDIT_BUREAU_HOUR'].isnull()


if treat_missing_values==1:
    
    before = len(df)
    print('\n Shape of df before dropping = ',df.shape)
    
    #values for age of car is assigned to -1 when one has no car
    #df.loc[df['FLAG_OWN_CAR'] == 'N', 'OWN_CAR_AGE'] = -1
    df['OWN_CAR_AGE'].fillna(0, inplace= True)
    
    #droped columns with missing value > 45% and are completely missing at ranom 
    df = cf.missing_value_analysis(df, 0.45)
    print('/n Shape of df after dropping = ',df.shape)
    
    #dropping as these are missing not at random
    df = df.dropna(axis=0, how='any', subset=['AMT_REQ_CREDIT_BUREAU_HOUR','OBS_60_CNT_SOCIAL_CIRCLE'], inplace=False)
    
    #filling missing value with np.nan for missing at random
    #df['OCCUPATION_TYPE'].fillna('unknown', inplace = True)
    
    #df.fillna(0, inplace = True)
     
    after = len(df)
    
    print("dropped rows %--->", round(1-(after/before),2)*100,"%")
    
    

if observed_corrections==1:
    
    print(df['CODE_GENDER'].value_counts())
    df = df[df['CODE_GENDER']!= 'XNA']
    
    print('Outliers analysis')
    vp.plot_outliers(df)
   
    #outlier treatment
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    df.drop(df[df['AMT_INCOME_TOTAL'] > 11000000].index, inplace = True)   
    df.drop(df[df['OBS_30_CNT_SOCIAL_CIRCLE'] > 30].index, inplace = True)   
    df.drop(df[df['DEF_30_CNT_SOCIAL_CIRCLE'] > 30].index, inplace = True)   
    df.drop(df[df['OBS_60_CNT_SOCIAL_CIRCLE'] > 300].index, inplace = True)   
    df.drop(df[df['DEF_60_CNT_SOCIAL_CIRCLE'] > 20].index, inplace = True)   
    df.drop(df[df['AMT_REQ_CREDIT_BUREAU_QRT'] > 200].index, inplace = True)   


if feature_engineering==1:
    
    num_df = df.select_dtypes(exclude='O')
 
    edu_dict = {'Incomplete higher':0,
                'Lower secondary':1,
                'Secondary / secondary special':2,
                'Higher education':3,
                'Academic degree':4
                }
    
    docs = [col for col in np.array(df.columns) if 'FLAG_DOCUMENT' in col ]
    
    possesions = [col for col in np.array(df.columns) if 'FLAG' in col and'FLAG_DOCUMENT' not in col]
    possesions = [ col for col in possesions if 'PHONE' in col or 'MOBIL' in col]
    
    live = [col for col in np.array(df.columns) if 'REGION_NOT' in col or 'CITY_NOT' in col]
    Soc_circle = [col for col in np.array(df.columns) if 'SOCIAL_CIRCLE' in col  and '30' in col]
    BUREAU_ENQ = [col for col in np.array(df.columns) if 'AMT_REQ_CREDIT_BUREAU' in col ]
    
    df['num_mean'] = df[num_df.columns].mean(axis=1)
    df['num_std'] = df[num_df.columns].std(axis=1)
    df['DAYS_BIRTH'] = df['DAYS_BIRTH']/-365
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED']*-1
    df['DAYS_REGISTERED'] = df['DAYS_REGISTRATION']*-1
    df['DAYS_ID_PUBLISH'] = df['DAYS_ID_PUBLISH']*-1
    df['NAME_EDUCATION_TYPE_NUM'] = df['NAME_EDUCATION_TYPE'].map(edu_dict)
    df['FE_FLAG_DOCUMENT_SUM'] = df[docs].std(axis=1)
    df['FE_FLAG_MOBIL_SUM'] = df[possesions].sum(axis=1)
    df['FE_LIVE_SUM'] = df[live].sum(axis=1)
    df['FE_BUREAU_ENQ'] = df[BUREAU_ENQ].sum(axis=1)
    df['FE_ANNUITY_CREDIT_RATIO'] = df['AMT_ANNUITY']/df['AMT_CREDIT']
    df['FE_GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE']/df['AMT_CREDIT'] 
    df['FE_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['FE_SOURCES_MEAN'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['FE_EXT_SIURCES_MULTIPLY'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['FE_CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    

if visual_analysis==1:
    # pl.visualize_y_vs_x(df,y_name)
    
    float_cols = []
    stack_cols = []
    
    for col in df.drop('SK_ID_CURR',axis=1).columns:
        if df[col].dtype == 'O' and len(df[col].unique())<10 :
                stack_cols.append(col)
                
    for col in df.drop('SK_ID_CURR',axis=1).columns:
        if df[col].dtype != 'O' and len(df[col].unique())>10 :
                float_cols.append(col)
    
    
    print('Data available for each class in categorical Variables')
    vp.stack_bar(df,stack_cols,y_name)
    
    print('numeric variables distribustion')
    vp.dist_bar(df,float_cols,y_name)
    


if merge_data :
    
    df = md.train_df(df) 
    df = df.drop([col for col in df.columns if 'SK_ID' in col],axis=1)
    df = cf.missing_value_analysis(df, 0.45)
    df_meta_desc = cf.primary_analysis(df)
    print('/n shape of the df : ',df.shape)
    
if skew_corrections==1:
    #df = pl.skew_correction(df)
    viusalise = 1
    dup_df = df.copy()
    i = 0
    
    print(''' \n \t \t skewness after transformation \n 
                 \t \t \t before correction \t after correction ''')

    for col in df.columns:
        if df[col].dtype != 'O' and len(df[col].unique())>10 :
            if (df[col].skew() > 0.9 or df[col].skew() < -0.9) and (min(df[col])>0):
                i+=1
                df[col] = np.log1p(df[col])
                print(col,' : ',round(dup_df[col].skew(),),' | ', df[col].skew())

                
                if viusalise:
                    fig, ax = plt.subplots(figsize=(15,8))
                    plt.subplot(1,2,1)
                    plt.title('Distribution before skew correction')
                    sns.kdeplot(data=dup_df, x=col, hue='TARGET')
            
                    plt.subplot(1,2,2)
                    plt.title('Distribution after skew correction')
                    sns.kdeplot(data=df, x=col, hue='TARGET')
                
                    plt.show ()  
               
#                 if  (df[col].skew() > 0.9 or df[col].skew() < -0.9):
               
# #                     print(col,f''' skew before correction : {dup_df[col].skew()}
# # skew after correction : {df[col].skew()} \n''')

                        
    print('skew corrected for Total number of columns : ',i)
    
    
                
if scaling_imputing ==1:
    
    num_df = df.select_dtypes(exclude='O') 
    num_df = num_df.drop('TARGET',axis=1)
    num_cols = num_df.columns
    
    # cat_df = df.select_dtypes(include='O')  
    # cat_cols = cat_df.columns

    # imputer_num = SimpleImputer(strategy='median')
    # num_df = imputer_num.fit_transform(num_df)
    
    # imputer_cat = SimpleImputer(strategy='constant', fill_value='Missing')
    # cat_df = imputer_cat.fit_transform(cat_df)
    
    scalaed = StandardScaler()
    scaled_df = scalaed.fit_transform(num_df)   
    num_df = pd.DataFrame(scaled_df, columns=num_cols)
    
    # cat_df = pd.DataFrame(cat_df, columns=cat_cols)

    # for col in cat_df.columns:
    #     df[col] = cat_df[col]
        
    
    for col in num_df.columns:
        df[col] = num_df[col]
        


if encoding==1:
    
    print("Shape of the df before encoding = ",df.shape)
    df= cf.one_hot_enc(df, nan_as_category=False)
    print("Shape of the df after encoding = ",df.shape)


    
if define_variables==1:
    df = df.fillna(-999)
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    y = df[y_name]
    x = df.drop([y_name],axis=1)
    n_dim = x.shape[1]
    print(x.shape)
   

if reduce_dim==1:
    rx = pl.reduce_dimensions(x,30); #print(x.shape)
    x = pd.DataFrame(rx)
    print("transformed x:")
    print(x.shape); print("")
  
x_train,x_test,y_train,y_test=ms.train_test_split(x,y,stratify=y ,test_size=0.3, random_state=89)

if oversample==1:
    #for only imbalanced data

    # x = x.fillna(-999)
    x_train,y_train = pl.oversampling(x_train,y_train)
    x_train = pd.DataFrame(x_train)
    print(x_train.shape)
    print(y_train.shape)
    print(y_train.value_counts())
    
df = cf.convert_types(df,print_info = True)
    
    
if compare==1:
    
    #compare models on sample
    n_samples = 5000
    #y_df = y.reshape(-1,1)
    df_temp = pd.concat((x_train,y_train),axis=1)
    
    df_sample = pl.stratified_sample(df_temp, y_name, n_samples)
    
    print("stratified sample:")
    print(df_sample[y_name].value_counts())
    
    y_sample = df_sample[y_name]
    x_sample = df_sample.drop([y_name],axis=1)
    model_meta_data = pl.compare_models(x_sample, y_sample, 111)
    

    
if cross_validate==1:
    #deciding the random state
    best_model = cl.XGBClassifier()
    pl.kfold_cross_validate(best_model, x, y,100)




if train_classification==1:
    
    
    print("Shape of training dataset : ",x_train.shape)
    np.random.seed(89)

    
    model = LogisticRegression(random_state=89)
    model.fit(x_train, y_train)
        
    
    test_pred = model.predict(x_test)
    train_pred = model.predict(x_train)

    
    print("Training:")
    print(classification_report(y_train, train_pred))
    print("Testing:")
    print(classification_report(y_test, test_pred))
    print('ROC AUC score : ',roc_auc_score(y_test, test_pred))
    print('confusion matrix: /n',confusion_matrix(y_test, test_pred))
    
            
    ####confusion matrix#####
    
    cm_test=confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(6,6))
    plt.title('Confusion matrix on test data')
    sns.heatmap(cm_test, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    
    ###### AUC - ROC #######
    
    # Create train and test curve
    fpr_train, tpr_train, thresh_train = roc_curve(y_train, train_pred)
    fpr_test, tpr_test, thresh_test = roc_curve(y_test, test_pred)
    
    # Create the straight line (how the graph looks like if the model does random guess instead)
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs)
    

    
    # Plot the ROC graph
    plt.figure(figsize=(8,6))
    plt.title('ROC Curve')
    plt.plot(fpr_train, tpr_train, label='Train')
    plt.plot(fpr_test, tpr_test, label='Test')
    plt.plot(p_fpr, p_tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    
    


    
    if False:
        #recycle the models with most important features
        fi = model.feature_importances_; print("fi count-->", len(fi))
        fi_dict={}
        
        for i in range(x.shape[1]):  
            fi_dict[x.columns[i]] = fi[i]
        fi_dict = pl.sort_by_value(fi_dict)
        #print([k for k,v in fi_dict])
        
        feat_imp = pd.DataFrame(fi_dict, columns=['name','value'])
        
        plt.figure(figsize=(10,8))
        plt.barh(feat_imp['name'][:40],feat_imp['value'][:40])
        plt.title('Feature Importance')
        plt.show()
        
        
        

        
        
   
