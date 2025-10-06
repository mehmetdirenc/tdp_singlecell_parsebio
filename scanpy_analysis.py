import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import os
import scipy.io as sio

## General scanpy settings ##
sc.settings.verbosity = 1 # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=100, fontsize=10, dpi_save=300, figsize=(5,4), format='png')
sc.settings.figdir = '/home/ubuntu/volume_750gb/results/tdp_project/figures'

# main_data_folder =


# print(sc.settings.figdir, os.getcwd())