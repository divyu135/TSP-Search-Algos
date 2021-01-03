import numpy as np
import math
import pandas as pd
import time
from tkinter import Tk, Canvas
from random import randint
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

stack = []
def tsp(adj):
    N = len(adj)
    path = []
    cost = 0
    visited = [0]*(N+1)
    visited[0] = 1
    stack.append(0)
    ele, dst = 0, 0
    minn = float('inf')
    minFlag = False
    # print(0,end="\t")
    path.append(0)

    while stack:
        ele = stack[-1]
        i = 0
        minn = float('inf')
        while i < N:
            if adj[ele][i] > 1 and visited[i] == 0 :
                if minn > adj[ele][i]:
                    minn =adj[ele][i]
                    dst = i
                    minFlag = True
                    cost += minn

            i+=1
        if minFlag:
            visited[dst] = 1
            stack.append(dst)
            path.append(dst)
            # print(dst,end="\t")
            minFlag = False
            continue
        stack.pop()
    return path, cost

def plot_route(df1, path):
    m = Basemap(projection='lcc', resolution=None,
                width=5E6, height=5E6,
                lat_0=-15, lon_0=-56)
    
    plt.figure(figsize=(15,10))
    plt.axis('off')
    plt.title("Route")

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
