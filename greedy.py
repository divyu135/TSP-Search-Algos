import numpy as np
import math
import pandas as pd
import time
from tkinter import Tk, Canvas
from random import randint
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import os

os.environ['PROJ_LIB'] = "C:/Users/Sheena/anaconda3/Library/share"

def algorithm(cities):
	best_order = []
	best_length = float('inf')

	for i_start, start in enumerate(cities):
		order = [i_start]
		length = 0

		i_next, next, dist = get_closest(start, cities, order)
		length += dist
		order.append(i_next)

		while len(order) < cities.shape[0]:
			i_next, next, dist = get_closest(next, cities, order)
			length += dist
			order.append(i_next)

		#print(order)

		if length < best_length:
			best_length = length
			best_order = order
			
	return best_order, best_length

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
    return t1**2 + t2**2

        
def plot_route(df1, path):
    m = Basemap(projection='lcc', resolution=None,
                width=5E6, height=5E6,
                lat_0=-15, lon_0=-56)
    
    plt.figure(figsize=(20,30))
    plt.axis('off')
    plt.title(label = "Route - Greedy Algorithm", fontsize=30)
    
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

