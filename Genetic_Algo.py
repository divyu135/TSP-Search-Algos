import sys
# sys.path.append('C:/Users/dell/Downloads/TSP-GA-master/TSP-GA-master/src/')
# sys.path.append('C:/Users/dell/Downloads/TSP-GA-master/TSP-GA-master/')
# print(sys.path)

import random
import argparse
from datetime import datetime
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sys import maxsize
from time import time
from random import random, randint, sample
from haversine import haversine
from math import sin, cos, sqrt, atan2, radians

def get_genes_from(fn, sample_n=0):
    df = pd.read_csv(fn)
    # df.drop(df.columns[[0]], axis=1, inplace=True)
    # genes = [Gene(row['latitude'], row['longitude'])
    #           for _, row in df.iterrows()]
    genes = [Gene(row['city'],row['latitude'], row['longitude'])
              for _, row in df.iterrows()]
    
    
    return genes if sample_n <= 0 else sample(genes, sample_n)

class Gene:  # City
    # keep distances from cities saved in a table to improve execution time.
    __distances_table = {}

    def __init__(self, name, lat, lng):
        self.name = name
        self.lat = lat
        self.lng = lng
        
    def get_distance_to(self, dest):
        # origin = (self.lat, self.lng)
        # dest = (dest.lat, dest.lng)

        # forward_key = origin + dest
        # backward_key = dest + origin

        # if forward_key in Gene.__distances_table:
        #     return Gene.__distances_table[forward_key]

        # if backward_key in Gene.__distances_table:
        #     return Gene.__distances_table[backward_key]

        # dist = int(haversine(origin, dest))
        # Gene.__distances_table[forward_key] = dist
        
        # dist = (dest[0] - origin[0])**2 + (dest[1] - origin[1])**2
        # return dist
        lat1,lon1 = self.lat, self.lng
        lat2, lon2 = dest.lat, dest.lng
        R = 6373.0 #radius if earth

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        return distance


class Individual:  # Route: possible solution to TSP
    def __init__(self, genes):
        assert(len(genes) > 3)
        self.genes = genes
        self.__reset_params()

    def swap(self, gene_1, gene_2):
        self.genes[0]
        a, b = self.genes.index(gene_1), self.genes.index(gene_2)
        self.genes[b], self.genes[a] = self.genes[a], self.genes[b]
        self.__reset_params()

    def add(self, gene):
        self.genes.append(gene)
        self.__reset_params()

    @property
    def fitness(self):
        if self.__fitness == 0:
            self.__fitness = 1 / self.travel_cost  # Normalize travel cost
        return self.__fitness

    @property
    def travel_cost(self):  # Get total travelling cost
        if self.__travel_cost == 0:
            for i in range(len(self.genes)):
                origin = self.genes[i]
                if i == len(self.genes) - 1:
                    dest = self.genes[0]
                else:
                    dest = self.genes[i+1]

                self.__travel_cost += origin.get_distance_to(dest)

        return self.__travel_cost

    def __reset_params(self):
        self.__travel_cost = 0
        self.__fitness = 0


class Population:  # Population of individuals
    def __init__(self, individuals):
        self.individuals = individuals

    @staticmethod
    def gen_individuals(sz, genes):
        individuals = []
        for _ in range(sz):
            individuals.append(Individual(sample(genes, len(genes))))
        return Population(individuals)

    def add(self, route):
        self.individuals.append(route)

    def rmv(self, route):
        self.individuals.remove(route)

    def get_fittest(self):
        fittest = self.individuals[0]
        for route in self.individuals:
            if route.fitness > fittest.fitness:
                fittest = route

        return fittest


def evolve(pop, tourn_size, mut_rate):
    new_generation = Population([])
    pop_size = len(pop.individuals)
    elitism_num = pop_size // 2

    # Elitism
    for _ in range(elitism_num):
        fittest = pop.get_fittest()
        new_generation.add(fittest)
        pop.rmv(fittest)

    # Crossover
    for _ in range(elitism_num, pop_size):
        parent_1 = selection(new_generation, tourn_size)
        parent_2 = selection(new_generation, tourn_size)
        child = crossover(parent_1, parent_2)
        new_generation.add(child)

    # Mutation
    for i in range(elitism_num, pop_size):
        mutate(new_generation.individuals[i], mut_rate)

    return new_generation


def crossover(parent_1, parent_2):
    def fill_with_parent1_genes(child, parent, genes_n):
        start_at = randint(0, len(parent.genes)-genes_n-1)
        finish_at = start_at + genes_n
        for i in range(start_at, finish_at):
            child.genes[i] = parent_1.genes[i]

    def fill_with_parent2_genes(child, parent):
        j = 0
        for i in range(0, len(parent.genes)):
            if child.genes[i] == None:
                while parent.genes[j] in child.genes:
                    j += 1
                child.genes[i] = parent.genes[j]
                j += 1

    genes_n = len(parent_1.genes)
    child = Individual([None for _ in range(genes_n)])
    fill_with_parent1_genes(child, parent_1, genes_n // 2)
    fill_with_parent2_genes(child, parent_2)

    return child


def mutate(individual, rate):
    for _ in range(len(individual.genes)):
        if random() < rate:
            sel_genes = sample(individual.genes, 2)
            individual.swap(sel_genes[0], sel_genes[1])


def selection(population, competitors_n):
    return Population(sample(population.individuals, competitors_n)).get_fittest()


def run_ga(genes, pop_size, n_gen, tourn_size, mut_rate, verbose=1):
    population = Population.gen_individuals(pop_size, genes)
    history = {'cost': [population.get_fittest().travel_cost]}
    counter, generations, min_cost = 0, 0, maxsize

    if verbose:
        print("-- TSP-GA -- Initiating evolution...")

    start_time = time()
    while counter < n_gen:
        population = evolve(population, tourn_size, mut_rate)
        cost = population.get_fittest().travel_cost

        if cost < min_cost:
            counter, min_cost = 0, cost
        else:
            counter += 1

        generations += 1
        history['cost'].append(cost)

    total_time = round(time() - start_time, 6)

    if verbose:
        print("-- TSP-GA -- Evolution finished after {} generations in {} s".format(generations, total_time))
        print("-- TSP-GA -- Travelling cost {}".format(min_cost))

    history['generations'] = generations
    history['total_time'] = total_time
    history['route'] = population.get_fittest()

    return history

def plot(costs, individual):
    plt.figure(figsize=(20,30))
    # plt.subplot(121)
    # plot_ga_convergence(costs)

    # plt.subplot(122)    
    plot_route(individual)
    plt.show()

def plot_ga_convergence(costs):
    x = range(len(costs))
    plt.title("GA Convergence")
    plt.xlabel('generation')
    plt.ylabel('cost (KM)')
    plt.text(x[len(x) // 2], costs[0], 'min cost: {} KM'.format(costs[-1]), ha='center', va='center')
    plt.plot(x, costs, '-')


def plot_route(individual):
    m = Basemap(projection='lcc', resolution=None,
                width=5E6, height=5E6,
                lat_0=-15, lon_0=-56)

    plt.axis('off')
    plt.title(label = "Route - Genetic Algorithm", fontsize=30)

    for i in range(0, len(individual.genes)):
        x, y = m(individual.genes[i].lng, individual.genes[i].lat)

        plt.plot(x, y, 'ok', c='r', markersize=5)
        plt.text(x,y,individual.genes[i].name)
        if i == len(individual.genes) - 1:
            x2, y2 = m(individual.genes[0].lng, individual.genes[0].lat)
        else:
            x2, y2 = m(individual.genes[i+1].lng, individual.genes[i+1].lat)

        plt.plot([x, x2], [y, y2], 'k-', c='r')        

    
        
