#!/usr/bin/env python
# coding: utf-8

# In[71]:


# Author: Przemyslaw Musz
# Nicolaus Copernicus University
# Technical Physics

import time as tm
import argparse
import mdshare
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as mpl

# Get initial timestamp
start_tmstmp = tm.time()


# Creating parser for command line args
parser = argparse.ArgumentParser(description = 'Dimensionality reduction of tensor datasets.')

parser.add_argument('-d', type = int, default = 500,
                    help = 'Decimation ratio for reducing amount of input data (5..10000).')
parser.add_argument('-i', type = str, default = 'alanine-dipeptide-3x250ns-heavy-atom-distances.npz',
                    help = 'Input filename from molecular dynamics database on FU Berlin FTP server, otherwise a default example file will be used.')
parser.add_argument('-c', type = str, default = 'ocean',
                    help = 'Input name of desired colormap (ocean, viridis, gist_earth are supported).')
parser.add_argument('-a', type = str, default = 'PCA',
                   help = 'Input name of desired dimensionality reduction algorithm (PCA and TSNE are supported).')

# Fetch variables from parser
args = parser.parse_args()

# Argument validation
dec = args.d
file = args.i;
c = args.c
algorithm = args.a

# Parameters validation

# Tuple for color maps
cmaps = ("ocean","viridis","gist_earth")
if c not in cmaps:
    print('WARNING: colormap name not recognized; default colormap \'ocean\' will be used!')
    c = 'ocean'

# Checking decimation ratio
if (dec < 5) or (dec > 10000):
    dec = 500
    print('WARNING: decimation ratio out of valid range; default value 500 will be used!')

try:
    # Downloading data from database
    dataset = mdshare.fetch(file)
    # Data stacking
    with np.load(dataset) as f:
        X = np.vstack([f[key] for key in sorted(f.keys())])
        # Get timestamp for start of data processing
        proc_start_tmstmp = tm.time()


        if algorithm == 'TSNE':
            # t-distributed Stochastic Neighbor Embedding
            Y = TSNE(n_components = 3).fit_transform(X[::dec])
        elif algorithm == 'PCA':
            # Principal Component Analysis
            Y = PCA(n_components = 3).fit_transform(X[::dec])
        else:
            print("ERROR: An error occured during selection of reduciton method. Exiting...")
            exit(-1)
        
        # Scaling data to desired range
        Y[:, 0] = np.interp(Y[:, 0], (Y[:, 0].min(), Y[:, 0].max()), (-np.pi, np.pi)) 
        Y[:, 1] = np.interp(Y[:, 1], (Y[:, 1].min(), Y[:, 1].max()), (-np.pi, np.pi)) 
        Y[:, 2] = np.interp(Y[:, 2], (Y[:, 2].min(), Y[:, 2].max()), (-np.pi, np.pi)) 
        
        # Get timestamp for end of data processing
        proc_end_tmstmp = tm.time()

        # Generating scatterplot 
        mpl.scatter(Y[:, 0], Y[:, 1], c = Y[:,2], s = 5, alpha = 0.6, cmap = c )
        
        # Axes limits
        mpl.xlim(-np.pi, np.pi) 
        mpl.ylim(-np.pi, np.pi) 
        
        # Axes ticks
        mpl.xticks([-np.pi, 0, np.pi], ['-π', 0, 'π']) 
        mpl.yticks([-np.pi, 0, np.pi], ['-π', 0, 'π']) 
        
        # Scaling plot area
        mpl.axis('scaled')
        
        # Color legend drawing
        legend = mpl.colorbar() 
        legend.set_ticks([-np.pi, 0, np.pi]) 
        legend.set_ticklabels(['-π', 0, 'π']) 
        
        # Get final timestamp
        end_tmstmp = tm.time()

        # Print execution time info
        time_elapsed_whole = end_tmstmp - start_tmstmp
        time_elapsed_proc = proc_end_tmstmp - proc_start_tmstmp
        percent = 100*(time_elapsed_proc / time_elapsed_whole)
        print("Total time elapsed: {t:6.3f} seconds, {p:3.1f}% for data dimensionality reduction.\n".format(t = time_elapsed_whole, p = percent))
         
        # Showing scatterplot
        mpl.show()
        
        print('Exitning...')


except Exception as e:
        print('Exitning due to errors:')
        print(e)





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




