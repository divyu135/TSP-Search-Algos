import numpy as np
import math
import pandas as pd
import time
from tkinter import Tk, Canvas
from random import randint
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import math 
maxsize = float('inf') 
N = None
final_path = None 
visited = None 
final_res = maxsize 

def find_distance(cities):
	distances = []
	for c1 in cities:
		lst=[]
		for c2 in cities:
			dist = dist_squared(c1,c2)
			lst.append(dist)
		distances.append(lst)
	return distances

def copyToFinal(curr_path): 
    final_path[:N + 1] = curr_path[:] 
    final_path[N] = curr_path[0] 

def firstMin(adj, i): 
    min = maxsize 
    for k in range(N): 
        if adj[i][k] < min and i != k: 
            min = adj[i][k] 
  
    return min
  
def secondMin(adj, i): 
    first, second = maxsize, maxsize 
    for j in range(N): 
        if i == j: 
            continue
        if adj[i][j] <= first: 
            second = first 
            first = adj[i][j] 
  
        elif(adj[i][j] <= second and 
             adj[i][j] != first): 
            second = adj[i][j] 
  
    return second 
  

def TSPRec(adj, curr_bound, curr_weight,  
              level, curr_path, visited): 
    global final_res 
      
    if level == N: 
        
        if adj[curr_path[level - 1]][curr_path[0]] != 0: 

            curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]] 
            if curr_res < final_res: 
                copyToFinal(curr_path) 
                final_res = curr_res 
        return

    for i in range(N): 
          
        if (adj[curr_path[level-1]][i] != 0 and
                            visited[i] == False): 
            temp = curr_bound 
            curr_weight += adj[curr_path[level - 1]][i] 
  
            if level == 1: 
                curr_bound -= ((firstMin(adj, curr_path[level - 1]) + 
                                firstMin(adj, i)) / 2) 
            else: 
                curr_bound -= ((secondMin(adj, curr_path[level - 1]) +
                                 firstMin(adj, i)) / 2) 

            if curr_bound + curr_weight < final_res: 
                curr_path[level] = i 
                visited[i] = True
                  
                # call TSPRec for the next level 
                TSPRec(adj, curr_bound, curr_weight,  
                       level + 1, curr_path, visited) 
  
            # Else we have to prune the node by resetting  
            # all changes to curr_weight and curr_bound 
            curr_weight -= adj[curr_path[level - 1]][i] 
            curr_bound = temp 
  
            # Also reset the visited array 
            visited = [False] * len(visited) 
            for j in range(level): 
                if curr_path[j] != -1: 
                    visited[curr_path[j]] = True
  
# This function sets up final_path 
def TSP(adj): 
      
    curr_bound = 0
    curr_path = [-1] * (N + 1) 
    visited = [False] * N 
  
    # Compute initial bound 
    for i in range(N): 
        curr_bound += (firstMin(adj, i) + 
                       secondMin(adj, i)) 
  
    # Rounding off the lower bound to an integer 
    curr_bound = math.ceil(curr_bound / 2) 
  
    # We start at vertex 1 so the first vertex  
    # in curr_path[] is 0 
    visited[0] = True
    curr_path[0] = 0
  
    # Call to TSPRec for curr_weight  
    # equal to 0 and level 1 
    TSPRec(adj, curr_bound, 0, 1, curr_path, visited) 


def get_closest(city, cities, visited):
	best_distance = float('inf')

	for i, c in enumerate(cities):

		if i not in visited:
			distance = dist_squared(city, c)

			if distance < best_distance:
				closest_city = c
				i_closest_city = i
				best_distance = distance

	return i_closest_city, closest_city, best_distance

def dist_squared(c1, c2):
    t1 = c2[0] - c1[0]
    t2 = c2[1] - c1[1]
    d = math.sqrt(t1**2 +t2**2)
    return round(t1**2 + t2**2,3)

        
def plot_route(df1, path, title):
    m = Basemap(projection='lcc', resolution=None,
                width=5E6, height=5E6,
                lat_0=-15, lon_0=-56)
    
    plt.figure(figsize=(20,30))
    plt.axis('off')
    plt.title(label = title, fontsize=30)

    for i, item in enumerate(path):
        x, y = m(df1['longitude'][item], df1['latitude'][item])

        plt.plot(x, y, 'ok', c='r', markersize=5)
        plt.text(x,y,df1['city'][item])
        if i == len(path) - 1:
            x2, y2 = m(df1['longitude'][path[0]], df1['latitude'][path[0]])
        else:
            i = i+1
            nextItem = path[i]
            x2, y2 = m(df1['longitude'][nextItem], df1['latitude'][nextItem])

        plt.plot([x, x2], [y, y2], 'k-', c='r') 
    plt.show()

