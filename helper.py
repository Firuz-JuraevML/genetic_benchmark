from sklearn import datasets  
from ucimlrepo import fetch_ucirepo 
import pandas as pd   
from sklearn.preprocessing import LabelEncoder 

def load_dataset(db_name): 
    if db_name == "Iris (Sklearn)": 
        dataset = datasets.load_iris() 
    elif db_name == "Breast Cancer (Sklearn)": 
        dataset = datasets.load_breast_cancer() 
    elif db_name == "Digits (SKlearn)": 
        dataset = datasets.load_digits() 
    elif db_name == "Ionosphere (UCI)": 
        ionosphere = fetch_ucirepo(id=52) 
        # data (as pandas dataframes) 
        X = ionosphere.data.features 
        y = ionosphere.data.targets 
        y = y['Class'].map({'g': 0, 'b': 1}) 
        return X, y
    elif db_name == "SPECT Heart (UCI)": 
        spect_heart = fetch_ucirepo(id=95) 
        # data (as pandas dataframes) 
        X = spect_heart.data.features 
        y = spect_heart.data.targets
        return X, y 
    elif db_name == "MONK Problems (UCI)": 
        monk_s_problems = fetch_ucirepo(id=70) 
        # data (as pandas dataframes) 
        X = monk_s_problems.data.features 
        y = monk_s_problems.data.targets   
        return X, y 
    elif db_name == "User Knowledge Modeling (UCI)": 
        user_knowledge_modeling = fetch_ucirepo(id=257) 
        # data (as pandas dataframes) 
        X = user_knowledge_modeling.data.features 
        y = user_knowledge_modeling.data.targets  
        le = LabelEncoder()
        y['UNS'] = le.fit_transform(y['UNS'])
        return X, y 
    
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names) 
    df['target'] = dataset.target  

    X = df[dataset.feature_names] 
    y = df.target    

    return X, y 

