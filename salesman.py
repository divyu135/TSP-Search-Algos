import sys
import greedy as gd
import Genetic_Algo as ga
import numpy as np
import branch_bound as bb
import nearest_neigbour as nn
import bfs
import dfs
import helper as hp
import hill_climbing as hc
import astar 
from SimAnneal import SimAnneal
from time import time
from random import randint
import pandas as pd



def main():
	#loading data
    title = "Travelling Salesman Problem"
    print("\n"+title)
    print("\n Search algorithms")
    print(" 1: Genetic Algorithm \n 2: Greedy Algorithm \n 3: Branch and Bound \n" 
        + " 4. Nearest Neighbour \n 5: Breadth First Search \n 6: Depth First Search \n"
        + " 7. Hill Climbing \n 8: Astar Algorithm \n 9: Simulated Annealing \n")
    type = input("Choose the algorithm 1/2/3/4/5/6/7/8/9: ")
    
    if(type == '1'):
        genes = ga.get_genes_from('./cities.csv')
        #genes = utils.get_genes_from('E:/Masters studies/Intro to AI/Project/code/TSP_27Dec/TSP/tsp0100.txt')
    
        print("-- Running TSP-GA with {} cities --".format(len(genes)))
        history = ga.run_ga(genes, 100, 20, 50, 0.02, 1)
        
        print("-- Drawing Route --")
        ga.plot(history['cost'], history['route'])
    
        city_names = [] 
        for i in range(0,len(history['route'].genes)):
            city_names.append(history['route'].genes[i].name)
        city_names.append(history['route'].genes[0].name)
        print(city_names)   
        print("-- Done --")
            
    if(type == '2'):
        df = pd.read_csv("./cities.csv")
        df1 = df.copy()
        df.drop(df.columns[[0]], axis=1, inplace=True)
        columns_titles = ["longitude","latitude"]
        df=df.reindex(columns=columns_titles)
        cities = df.to_numpy()
        # print(cities)
        #calculating path
        start = time()
        path, length = gd.algorithm( cities )
        city_names = []
        for i in path:
            city_names.append(df1['city'][i])
        city_names.append(df1['city'][path[0]])
        print(city_names)
        tottime = time() - start
        length = hp.calculate_cost(path,df1)
        print( "Found path of length %s KM in %s seconds" % ( round(length,2), round(tottime, 2) ) )
        #displaying path        
        gd.plot_route(df1, path)
 
    if(type == '3'):
        df = pd.read_csv("./cities.csv")
        df1 = df.copy()
        df.drop(df.columns[[0]], axis=1, inplace=True)
        columns_titles = ["longitude","latitude"]
        df=df.reindex(columns=columns_titles)
        cities = df.to_numpy()
        
        start = time()
        adj = bb.find_distance( cities[:10] )
        bb.N = len(cities[:10])
        bb.final_path = [None] * (bb.N + 1) 
        bb.visited = [False] * bb.N 

        bb.TSP(adj) 
        path = bb.final_path
        city_names = []
        for i in path:
            city_names.append(df1['city'][i])
        city_names.append(df1['city'][path[0]])
        print(city_names)
        tottime = time() - start
        length = hp.calculate_cost(path,df1)
        print( "Found path of length %s KM in %s seconds" % ( round(length,2), round(tottime, 2) ) )     
        bb.plot_route(df1, path, "Route - Branch & Bound")
    
    if(type == '4'):
        df = pd.read_csv("./cities.csv")
        df1 = df.copy()
        df.drop(df.columns[[0]], axis=1, inplace=True)
        columns_titles = ["longitude","latitude"]
        df=df.reindex(columns=columns_titles)
        cities = df.to_numpy()
        start = time()
        adj = bb.find_distance( cities )
        path, length = nn.tsp(adj)
        city_names = []
        for i in path:
            city_names.append(df1['city'][i])
        city_names.append(df1['city'][path[0]])
        print(city_names)
        tottime = time() - start
        length = hp.calculate_cost(path,df1)
        print( "Found path of length %s KM in %s seconds" % ( round(length,2), round(tottime, 2) ) )     
        bb.plot_route(df1, path,"Route - Nearest Neighbour")
    
    if(type == '5'):
        df = pd.read_csv("./cities.csv")
        df1 = df.copy()
        df.drop(df.columns[[0]], axis=1, inplace=True)
        columns_titles = ["longitude","latitude"]
        df=df.reindex(columns=columns_titles)
        cities = df.to_numpy()
        start = time()
        adj = bb.find_distance( cities[:8] )

        bfs.graph = adj
        path, length = bfs.algorithm()
        city_names = []
        for i in path:
            city_names.append(df1['city'][i])
        city_names.append(df1['city'][path[0]])
        print(city_names)
        tottime = time() - start
        length = hp.calculate_cost(path,df1)
        print( "Found path of length %s KM in %s seconds" % ( round(length,2), round(tottime, 2) ) )     
        bb.plot_route(df1, path,"Route - Breadth First Search")

    if(type == '6'):
        df = pd.read_csv("./cities.csv")
        df1 = df.copy()
        df.drop(df.columns[[0]], axis=1, inplace=True)
        columns_titles = ["longitude","latitude"]
        df=df.reindex(columns=columns_titles)
        cities = df.to_numpy()
        start = time()
        adj = bb.find_distance( cities[:8] )

        dfs.graph = adj
        path, length = dfs.algorithm()
        city_names = []
        for i in path:
            city_names.append(df1['city'][i])
        city_names.append(df1['city'][path[0]])
        print(city_names)
        tottime = time() - start
        length = hp.calculate_cost(path,df1)
        print( "Found path of length %s KM in %s seconds" % ( round(length,2), round(tottime, 2) ) )     
        bb.plot_route(df1, path,"Route - Depth First Search")
    
    if(type == '7'):
        df = pd.read_csv("./cities.csv")
        df1 = df.copy()
        df.drop(df.columns[[0]], axis=1, inplace=True)
        columns_titles = ["longitude","latitude"]
        df=df.reindex(columns=columns_titles)
        cities = df.to_numpy()
        start = time()
        adj = bb.find_distance( cities )
        N =len(cities)

        problem = hc.TspProblem(N,adj)
        path = hc.hill_climbing_random_restarts(problem, restarts_limit=200).path()[0][1][0][0:-1]
        city_names = []
        for i in path:
            city_names.append(df1['city'][i])
        city_names.append(df1['city'][path[0]])
        print(city_names)
        tottime = time() - start
        length = hp.calculate_cost(path,df1)
        print( "Found path of length %s KM in %s seconds" % ( round(length,2), round(tottime, 2) ) )     
        bb.plot_route(df1, path,"Route - Hill Climbing")
        
    if(type == '8'):
        df = pd.read_csv("./cities.csv")
        df1 = df.copy()
        df.drop(df.columns[[0]], axis=1, inplace=True)
        columns_titles = ["longitude","latitude"]
        df=df.reindex(columns=columns_titles)
        cities = df.to_numpy()
        temp = cities[:10].copy()
        start = time()

        astar.astar_main_run(temp)
        path = astar.path
        
        city_names = []
        for i in path:
            city_names.append(df1['city'][i])
        city_names.append(df1['city'][path[0]])
        print(city_names)
        tottime = time() - start
        length = hp.calculate_cost(path,df1)
        print( "Found path of length %s KM in %s seconds" % ( round(length,2), round(tottime, 2) ) )     
        bb.plot_route(df1, path,"Route - A* Algorithm")

    if(type == '9'):
        df = pd.read_csv("cities.csv", usecols = ["city","latitude","longitude"])
        coords =[]
        for i in range(df.shape[0]):
            coords.append([i,df["latitude"].iloc[i],df["longitude"].iloc[i]])
        df = pd.read_csv("cities.csv")
        df1 = df.copy()
        start = time()

        sa = SimAnneal(coords, stopping_iter=5000)
        sa.anneal()

        path = sa.best_solution
        city_names = []
        for i in path:
            city_names.append(df1['city'][i])
        city_names.append(df1['city'][path[0]])
        print(city_names)
        tottime = time() - start
        length = hp.calculate_cost(path,df1)
        print( "Found path of length %s KM in %s seconds" % ( round(length,2), round(tottime, 2) ) ) 
        bb.plot_route(df1, sa.best_solution,"Route - Simulated Annealing")
        # sa.plot_learning()

if __name__ == "__main__":
	main()
