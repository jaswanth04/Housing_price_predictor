from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_model():
    filename = "model/finalized_model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def pre_processing():
    col_names = ["status", "duration", "credit_history", "purpose", "credit_amount",
            "savings_account", "employed_since", "installment_rate", "maritial_status_sex",
            "other_debtors", "resident_since", "property", "age", "other_installments",
            "housing", "existing_credits", "job", "no_of_dependents", "telephone", "foreign_worker", "credit"]
    df = pd.read_csv("data/german.data", sep=" ", header=None, names=col_names)
    
    targetcols=['credit']
    #Target Variable Encoding
    df['credit'].replace({1:0,2:1},inplace=True) 
    
    #Out of categorical ,below are ordinal variable
    #Ordinal variable Encoding
    df['employed_since'].replace({'A71':0,'A72':1,'A73':2,'A74':3,'A75':4},inplace=True)
    df['savings_account'].replace({'A65':0,'A61':1,'A62':2,'A63':3,'A64':4},inplace=True)
    
    # Dummy variables/One hot encoding
    catcols=[col for col in df.columns if (df[col].dtype not in ['float64','int64','uint8'])\
             & (col not in targetcols)]
    numcols=[col for col in df.columns if (df[col].dtype in ['float64','int64','uint8'])\
            & (col not in targetcols)]
    catcoldumy=pd.get_dummies(df[catcols])
    df=df.drop(catcols,axis=1)
    df = df.merge(catcoldumy, left_index=True, right_index=True)
    
    corr_m=df.corr()   
    corr_m['abs_credit']=corr_m['credit'].abs()
    #Taking Highly Correlated Variable with target 
    corr_cols=corr_m[corr_m['abs_credit']>0.02].index.to_list()
    corr_cols.remove('credit')
    xcols=corr_cols
    
    #Divison of X and Y Variable
    y=df[targetcols].values
    X=df[xcols].values
    y=y.reshape(len(y),)
    #Oversampling
    sm = SMOTE(sampling_strategy='auto')
    X, y = sm.fit_sample(X, y)
    return(X,y)

    
def preprocess_predict(sample_query_dict):
    data=[sample_query_dict]
    new_data=pd.DataFrame(data,index=range(0,len(data)))
    df=new_data.copy()
    df['employed_since'].replace({'A71':0,'A72':1,'A73':2,'A74':3,'A75':4},inplace=True)
    df['savings_account'].replace({'A65':0,'A61':1,'A62':2,'A63':3,'A64':4},inplace=True)
    
    targetcols=['credit']
    catcols=[col for col in df.columns if (df[col].dtype not in ['float64','int64','uint8'])\
             & (col not in targetcols)]
    numcols=[col for col in df.columns if (df[col].dtype in ['float64','int64','uint8'])\
            & (col not in targetcols)]
    catcoldumy=pd.get_dummies(df[catcols])
    df=df.drop(catcols,axis=1)
    df = df.merge(catcoldumy, left_index=True, right_index=True)
    filename = 'model/xcols.sav'
    xcols = pickle.load(open(filename, 'rb'))
    for remcols in set(xcols)-set(df.columns):
        df[remcols]=0
    return df[xcols].values
    

def explain_model():
    clf = load_model()
    xcols = ['duration', 'credit_amount', 'savings_account', 'employed_since', 'installment_rate', 'age', 
    'existing_credits', 'status_A11', 'status_A12', 'status_A13', 'status_A14', 'credit_history_A30', 'credit_history_A31',
     'credit_history_A32', 'credit_history_A34', 'purpose_A40', 'purpose_A41', 'purpose_A410', 'purpose_A42', 'purpose_A43', 
     'purpose_A45', 'purpose_A46', 'purpose_A48', 'purpose_A49', 'maritial_status_sex_A91', 'maritial_status_sex_A92', 'maritial_status_sex_A93', 
     'other_debtors_A102', 'other_debtors_A103', 'property_A121', 'property_A124', 'other_installments_A141', 'other_installments_A142', 
     'other_installments_A143', 'housing_A151', 'housing_A152', 'housing_A153', 'job_A172', 'job_A174', 'telephone_A191', 'telephone_A192', 
     'foreign_worker_A201', 'foreign_worker_A202']
    fig = plt.figure(figsize = (15, 35))
    ax = fig.add_subplot(1, 1, 1)
    coeff = pd.DataFrame({"feature":xcols,
                      "importance":np.transpose(clf.coef_[0])})

    coeff.sort_values(by = "importance").set_index("feature").plot.barh(title = "Feature Importance of Linear Model (LR)", color="chocolate", ax=ax)
    
    return fig

def predict_credit(query_dict):
    
    feature_vector = preprocess_predict(query_dict)
    print(feature_vector)

    model = load_model()


    prediction = model.predict(feature_vector)

    if prediction == 1:
        return "Bad"
    else:
        return "Good"
    
def retrain(extra_X, extra_y):
    clf = LogisticRegression(penalty='l2',C=1.0, max_iter=10000,solver='liblinear', class_weight='balanced')
    X, y = pre_processing()

    print(X.shape)
    print(y.shape)

    new_X = preprocess_predict(extra_X)
    new_y = 0 if extra_y == "good" else 1

    print(new_X)
    print(len(new_X))
 
    X = np.append(X, [new_X[0]], axis=0)
    y = np.append(y, new_y)

    print(X.shape)
    print(y.shape)

    clf.fit(X, y)

    filename = 'model/finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    return "Model successfully refreshed"



# Trying the utils file

# sample_query_dict = {'status': 'A13', 'duration': 4, 'credit_history': 'A31', 'purpose': 'A41', 'credit_amount': 1000, 'savings_account': 'A62', 'employed_since': 'A72', 'installment_rate': 4, 'maritial_status_sex': 'A92', 'other_debtors': 'A103', 'resident_since': 3, 'property': 'A122', 'age': 24, 'other_installments': 'A142', 'housing': 'A151', 'existing_credits': 4, 'job': 'A171', 'no_of_dependents': 2, 'telephone': 'A192', 'foreign_worker': 'A202'}


# print(predict_credit(sample_query_dict))

# # fig = explain_model()
# # plt.show()

# retrain(sample_query_dict, "bad")