# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:57:43 2018

@author: Robin
"""

import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as pl
import seaborn as sns
import os


os.chdir('C:\\Users\\Robin\\Desktop\\MA_Code\\COMPAS')

# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986
mapdata = np.loadtxt("SanFranMap.txt")
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

z = zipfile.ZipFile('C:\\Users\\Robin\\Desktop\\MA_Code\\COMPAS\\all.zip')
train = pd.read_csv(z.open('train.csv'))

#Get rid of the bad lat/longs
train['Xok'] = train[train.X<-121].X
train['Yok'] = train[train.Y<40].Y
train = train.dropna()
trainP = train[train.Category == 'PROSTITUTION'] #Grab the prostitution crimes
train = train[1:300000] #Can't use all the data and complete within 600 sec :(

#Seaborn FacetGrid, split by crime Category
g= sns.FacetGrid(train, col="Category", col_wrap=6, size=5, aspect=1/asp)

#Show the background map
for ax in g.axes:
    ax.imshow(mapdata, cmap=pl.get_cmap('gray'), 
              extent=lon_lat_box, 
              aspect=asp)
#Kernel Density Estimate plot
g.map(sns.kdeplot, "Xok", "Yok", clip=clipsize)

pl.savefig('category_density_plot.png')

#Do a larger plot with prostitution only
pl.figure(figsize=(20,20*asp))
ax = sns.kdeplot(trainP.Xok, trainP.Yok, clip=clipsize, aspect=1/asp)
ax.imshow(mapdata, cmap=pl.get_cmap('gray'), 
              extent=lon_lat_box, 
              aspect=asp)
pl.savefig('prostitution_density_plot.png')



'''
#Do a heatmap for visualization
a = pd.concat([trainP.Xok, trainP.Yok], axis = 1)
b = np.array(a)

pl.figure(figsize=(20,20*asp))
heat_map = sns.heatmap(b, linewidths = .5)
heat_map.imshow(mapdata, cmap=pl.get_cmap('gray'), 
              extent=lon_lat_box, 
              aspect=asp)

pl.savefig('prostitution_heat_map_plot.png')


a = np.linspace(min(trainP.Xok), max(trainP.Xok), num = 50)
b = np.linspace(min(trainP.Yok), max(trainP.Yok), num = 50)

data = pd.DataFrame(data = 0, index = a, columns = b)



for x,y in zip(trainP.Xok.values, trainP.Yok.values):
    #x = trainP.Xok.values[0], y = trainP.Yok.values[0]
    for x1, y1 in zip(data.index.values, data.columns.values):
        #x1 = data.index.values[0], y1 = data.columns.values[0]
            if str(x)[:5] == str(x1)[:5] and str(y)[:5] == str(y1)[:5]:
                data.at[x1, y1] += 1
'''





