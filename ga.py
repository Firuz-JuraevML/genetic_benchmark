import math 
import pandas as pd 
from collections import namedtuple
from random import choices, randint, randrange, random, sample
from typing import List, Callable, Tuple 
from functools import partial
from sklearn.metrics import (accuracy_score, f1_score)

from diversity import *  

import streamlit as st 

Genome = List[int] 
Population = List[Genome] 
BaseClassifier = namedtuple('BaseClassifier', ['name', 'model'])  


def get_initial_population(size: int, length: int): 
    population = [] 
    for i in range(size): 
        population.append(choices([0, 1], k=length)) 

    return population 


class GeniticAlgorithm: 
    def __init__(self, pool: [BaseClassifier], metric_function: accuracy_score, ensemble_model, 
                 X_dsel, y_dsel, X_test, y_test): 
        self.pool = pool
        self.metric_function = metric_function 
        self.ensemble_model  = ensemble_model 
        self.X_dsel = X_dsel 
        self.y_dsel = y_dsel 
        self.X_test = X_test 
        self.y_test = y_test 

        self.measure_diversity()
        

    def measure_diversity(self):  
        prediction_dict = {} 
    
        for bc in self.pool: 
            preds = bc.model.predict(self.X_dsel) 
            prediction_dict[bc.name] = preds
    
        # Calculate diversity 
        df_diversity = pd.DataFrame(columns=['model'] + list(prediction_dict.keys())) 
        
        for main_model in list(prediction_dict.keys()): 
            row_dict = {"model": main_model} 
            
            for second_model in list(prediction_dict.keys()): 
                row_dict[second_model] = disagreement_measure(self.y_dsel.to_numpy(), prediction_dict[main_model], prediction_dict[second_model]) 
    
            df_diversity = pd.concat([df_diversity, pd.DataFrame([row_dict])], ignore_index=True) 
                
        # dictionary 
        diversity_dictionary = {}

        for first_model in df_diversity.model.to_list(): 
            for second_model in df_diversity.model.to_list(): 
                diversity_dictionary[f"{first_model}-{second_model}"] = df_diversity[df_diversity.model == first_model][second_model].iloc[0] 
        
        self.diversity_dictionary = diversity_dictionary 

    
    def genome_to_pool(self, genome: Genome): 
        result = [] 
        for i, model in enumerate(self.pool): 
            if genome[i] == 1: 
                result += [model.name] 
    
        return result 

    
    def generate_genome(self, length: int, initial: str='random') -> Genome: 
        if initial == 'random': 
            return choices([0, 1], k=length) 
        elif initial == 'all': 
            return choices([1], k=length)  
        elif initial == 'half': 
            p1 = int(length/2) 
            p2 = length - p1 
            return choices([1], k=p1) + choices([0], k=p2)  
            
    
    
    def generate_population(self, size: int, genome_length: int, initial_population: str) -> Population: 
        return [self.generate_genome(genome_length, initial=initial_population) for _ in range(size)] 


    def diversity_fitness(self, genome: Genome) -> float: 
        models = [self.pool[i].model for i, value in enumerate(genome) if value == 1]  
        model_names = [self.pool[i].name for i, value in enumerate(genome) if value == 1]  

        if len(models) <= 1: 
            return 1 

        diversity_list = [] 
        for first_model in model_names: 
            for second_model in model_names: 
                diversity_list.append(self.diversity_dictionary[f"{first_model}-{second_model}"]) 

        return round(sum(diversity_list)/len(diversity_list), 4) 
        
    
    def fitness(self, genome: Genome) -> float: 
        models = [self.pool[i].model for i, value in enumerate(genome) if value == 1] 
        # print(self.genome_to_pool(genome, pool)) 
        
        if len(models) <= 1: 
            return 0 
        
        ensemble = self.ensemble_model(models)
        ensemble.fit(self.X_dsel, self.y_dsel)
    
        preds = ensemble.predict(self.X_test) 
        score = self.metric_function(self.y_test, preds) 
    
        return score
    
    
    def selection_pair(self, population: Population, fitness_func) -> Population: 
        return choices(
            population=population, 
            weights=[fitness_func(genome) for genome in population], 
            k=2
        )
    
    
    def single_point_crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]: 
        if len(a) != len(b): 
            raise ValueError("Genome a and b must be the same length")
    
    
        length = len(a) 
        if length < 2: 
            return a, b 
    
        p = randint(1, length - 1) 
        return a[0:p] + b[p:], b[0:p] + a[p:] 


    def two_points_crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        if len(a) != len(b): 
            raise ValueError("Genome a and b must be the same length") 

        length = len(a) 
        if length < 2: 
            return a, b  

        p1 = randint(1, int(length/2)) 
        p2 = randint(int(length/2), length - 1) 

        return a[0:p1] + b[p1:p2] + a[p2:], b[0:p1] + a[p1:p2] + b[p2:] 


    def uniform_crossover(self, a: Genome, b: Genome) -> Tuple[Genome, Genome]: 
        if len(a) != len(b): 
            raise ValueError("Genome a and b must be the same length")  

        length = len(a) 
        if length < 2: 
            return a, b 

        for i in range(0, length, 2): 
            a[i], b[i] = b[i], a[i]  

        return a, b 
        
    
    def mutation(self, genome: Genome, num: int = 3, probability: float = 0.5) -> Genome: 
        for _ in range(num): 
            index = randrange(len(genome)) 
            genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    
        return genome 


    def swap_mutation(self, genome: Genome): 
        length = len(genome)  
        
        p1 = randint(1, int(length/2)) 
        p2 = randint(int(length/2), length - 1)  

        genome[p1], genome[p2] = genome[p2], genome[p1] 

        return genome  


    def tournament_selection(self, population: Population) -> Genome: 
        random_population = sample(population, k=5)  
        # random_population = sorted(random_population, key=lambda genome: fitness_func(genome), reverse=True) 

        return random_population[0]
        

    def run_evolution(self, 
        fitness_limit: int,
        generation_limit: int = 100, 
        initial_population: str = 'random', 
        seed_population: Population = None, 
        parent_selection: str = 'diversity', 
        crossover: str = 'two_points') -> Tuple[Population, int]:

        # population = self.generate_population(size=10, genome_length=len(self.pool), initial_population=initial_population)
        population = seed_population 
        
        if crossover == 'single_point': 
            crossover_func = self.single_point_crossover 
        elif crossover == 'two_points': 
            crossover_func = self.two_points_crossover 
        elif crossover == 'uniform':
            crossover_func = self.uniform_crossover 
    
        for i in range(generation_limit):
            population = sorted(population, key=lambda genome: self.fitness(genome), reverse=True)
            
            st.write(self.genome_to_pool(population[0]))                  # <----- Printing Genome 
            st.write(f"Best Score: {self.fitness(population[0]):.3f}")    # <----- Printing Best Score   
    
            if self.fitness(population[0]) >= fitness_limit:
                break

            if parent_selection == 'rank': 
                next_generation = population[0:2] # <- selecting top two for parenting  
            elif parent_selection == 'diversity': 
                next_generation = [population[0]]
                d_population = sorted(population, key=lambda genome: self.diversity_fitness(genome), reverse=True) 
                next_generation.append(d_population[0])
            elif parent_selection == 'tournament': 
                next_generation = [population[0]] 
                next_generation.append(self.tournament_selection(population[2:])) # <- selecting with tournament selection 

            parent_1 = next_generation[0] 
            parent_2 = next_generation[1] 

    
            for j in range(int(len(population) / 2) - 1):
                # parents = self.selection_pair(population, self.fitness)
                # offspring_a, offspring_b = crossover_func(parents[0], parents[1])
                offspring_a, offspring_b = crossover_func(parent_1[:], parent_2[:]) 
                offspring_a = self.mutation(offspring_a)
                offspring_b = self.swap_mutation(offspring_b)
                next_generation += [offspring_a, offspring_b]
    
            population = next_generation
    
        return population, i