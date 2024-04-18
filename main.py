import streamlit as st
from PIL import Image 
import pandas as pd 
import datetime
import numpy as np 
from collections import namedtuple
import time 

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

from deslib.des.des_p import DESP
from deslib.des.knop import KNOP 

from sklearn.model_selection import train_test_split 

from sklearn.metrics import (accuracy_score, f1_score)

from sklearn import datasets   
import helper 
import ga


def main(): 
    info_msg = """Genetic Algorithm (GA) is a search-based optimization technique based on the principles of Genetics and Natural Selection. 
                  It is frequently used to find optimal or near-optimal solutions to difficult problems which otherwise would take a lifetime to solve. 
                  It is frequently used to solve optimization problems, in research, and in machine learning."""
    
    st.markdown('<h3 style="color:#4B7CA7;font-size:24;">🧬 Genetic Algorithm based Dynamic Ensemble Selection</h3>', unsafe_allow_html=True)  
    st.sidebar.markdown('<h3 style="color:#4B7CA7;font-size:16;">⚙️ Settings</h3>', unsafe_allow_html=True)
    st.info(info_msg, icon="ℹ️") 

    open_dataset = st.sidebar.selectbox('Select dataset: ',('Iris (Sklearn)', 'Ionosphere (UCI)', 'SPECT Heart (UCI)', 'MONK Problems (UCI)', 
                                                            'User Knowledge Modeling (UCI)', 'Breast Cancer (Sklearn)', 'Digits (SKlearn)')) 
    des  = st.sidebar.selectbox('Ensemble Model: ',('DESP', 'KNORAE', 'KNORAU'))  
    pool = st.sidebar.multiselect('Base Models: ',['XGB', 'LGBM', 'RF', 'KNN', 'DT', 'SVC', 'MLP'], 
                                     ['XGB', 'LGBM', 'RF', 'KNN', 'DT', 'SVC', 'MLP']) 
    
    initial_population = st.sidebar.selectbox('Select initial population: ', ('all', 'random', 'half')) 
    parent_select      = st.sidebar.selectbox('Select parent selection: ', ('diversity', 'rank', 'tournament'))
    crossover          = st.sidebar.selectbox('Select crossover technique: ', ('two_points', 'single_point', 'uniform')) 
    mutation           = st.sidebar.selectbox('Select mutation technique: ', ('swap', 'bit-flip'))  
    generation_limit   = st.sidebar.number_input("Generations limit: ", 0, 200, 30) 

    submit_btn = st.sidebar.button("🔬 Run Experiment")

    if submit_btn: 
        st.toast("⏳ Dataset is loading ...")
        with st.status("Downloading data..."): 
            X, y = helper.load_dataset(open_dataset) 
            st.write(f"Dataset: **{open_dataset}**")
            st.write(f"Dataset shape: {X.shape}")
        
        # Dataset split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42) 
        X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25, random_state=42) 

        BaseClassifier = namedtuple('BaseClassifier', ['name', 'model'])  

        pool = [
            BaseClassifier('XGB', XGBClassifier(random_state=42)), 
            BaseClassifier('MLP', MLPClassifier(random_state=42)), 
            BaseClassifier('KNN', KNeighborsClassifier()),  
            BaseClassifier('XGB2', XGBClassifier(learning_rate=0.1, n_estimators=250, random_state=42)),  
            BaseClassifier('RF', RandomForestClassifier(random_state=42)), 
            BaseClassifier('RF2', RandomForestClassifier(n_estimators=230, random_state=42)), 
            BaseClassifier('LR', LogisticRegression(random_state=42)), 
            BaseClassifier('SVC', SVC(random_state=42)), 
            BaseClassifier('DT', DecisionTreeClassifier(max_depth=10, random_state=42)),
            BaseClassifier('DT2', DecisionTreeClassifier(max_depth=5, random_state=42)),
            BaseClassifier('LGBM', LGBMClassifier(random_state=42)), 
            BaseClassifier('LGBM2', LGBMClassifier(learning_rate=0.05, n_estimators=250, random_state=42)), 
        ]

        st.toast("⏳ Training base classifiers ...")
        with st.status("Training base classifiers ..."):  
            for base_classifier in pool:  
                base_classifier.model.fit(X_train, y_train) 
        

        seed_population = ga.get_initial_population(10, len(pool)) 

        genetic = ga.GeniticAlgorithm(pool, accuracy_score, DESP, X_dsel, y_dsel, X_test, y_test) 

        start_time = time.time()   
        st.toast("💡 Genetic Algorithm searching ...") 
        with st.status("Genetic Algorithm searching  ..."):  
            population, generations = genetic.run_evolution(
                fitness_limit=1.0, 
                generation_limit=generation_limit, 
                initial_population=initial_population, 
                seed_population = seed_population, 
                parent_selection=parent_select, 
                crossover=crossover
            )
        end_time = time.time() 
        
        st.write("⏱ Time (m): {}".format(round((end_time - start_time)/60, 2)))  
        st.write(f"🧬 Best Genome: {genetic.genome_to_pool(population[0])}")
        st.write(f"🥇 Best Score: {genetic.fitness(population[0]):.3f}")

        st.markdown("***")
        seeds = [10, 42, 45, 52, 56, 72, 84, 91, 93, 99]
        scores = [] 

        st.code(f"🧬 Best Genome: {genetic.genome_to_pool(population[0])}")
        st.code(f"Seeds: {seeds}")

        for seed in seeds: 
            # Dataset split
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=seed) 
            X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25, random_state=42)   

            for base_classifier in pool:  
                base_classifier.model.fit(X_train, y_train)  

            genetic = ga.GeniticAlgorithm(pool, accuracy_score, DESP, X_dsel, y_dsel, X_test, y_test) 

            score = genetic.fitness(population[0])
            st.code(f"Seed {seed} => Score: {score:.3f}") 

            scores.append(score) 

        scores_np = np.array(scores)
 
        st.write(f"Mean score: **{scores_np.mean():.3f}** ± (std: **{scores_np.std():.3f}**)") 







if __name__ == '__main__':
    main()  