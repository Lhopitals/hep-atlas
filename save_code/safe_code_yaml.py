#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import sys
import yaml
import awkward as ak
import vector

path = '/home/gabbo/hep-atlas/data/' 

#idea
# 0.25 met 0.5   [["met", 0.25, 0.5], ["mjj", 0.5, "empty"]]
# mjj 0.5
# cambiar todo a una unica lista con 3 columnas
# si no hay nada usar np.nan o np.inf

##############################Funciones que ocuparemos#################
# Leer archivos de data_yamluración .yaml
def read_data_yaml(data_yaml_file):
    with open(data_yaml_file) as f:
        data_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return data_yaml

#Leer archivos root
def read_root_file(path, filename, tree_name):
    file = uproot.open(path + filename)
    tree = file[tree_name]
    return tree

#Se crean variables con lo recuperado del archivo .yaml
data_yaml = read_data_yaml('previous_data.yaml')
#items= data_yaml['datasets'].items() #esto devuelve todas las claves y valores del diccionario
datasets = data_yaml['datasets'].values() #esto devuelve solo los valores de cada variable.
variables = data_yaml['recover_branches']
#print(variables)

# crea un dataframe con todos los datos
def read_datasets(datasets, variables):
    df_all = pd.DataFrame()
    for data in datasets:
        datos = read_root_file(path, data, "miniT")
        df_data = datos.arrays(variables, library="pd")    
        df_data = df_data[df_data['mjj'] > np.nan] 
        #print(data)
        #print(df_data.describe())

    return df_all



def do_cut(df, variable_corte, valor_corte, up = True):
    if up:
        return df[df[variable_corte] < valor_corte]
    if not up:
        return df[df[variable_corte] > valor_corte]
cut = data_yaml['cuts'].items()
print(cut)

def do_cuts(df):
    df_with_cuts = df
    
    # for cut in cuts_up:
        #     df_data = do_cut(df_data, cut[0], cut[1], up = True)
        
        # for cut in cuts_down:
        #     df_data = do_cut(df_data, cut[0], cut[1], up = False)

    return df_with_cuts



# read_root_file(path, 'signal/frvz_vbf_500757.root', 'miniT')
read_datasets(datasets, variables)


##########################################################
############## SIGNIFICANCIA Y CORTES ####################
##########################################################

# SIGNIFICANCE DEFINITION
def significance(signal: float,background: float) -> float:
    return np.sqrt(2 * abs( (signal+background) * np.log(1 + (signal/background)) - signal))

df_calculo_significancia = pd.DataFrame()


# data_yaml = read_data_yaml('previous_data.yaml')
# test_1 = data_yaml['datasets']
# test_2 = data_yaml['cuts']['max_dphijj']
# print(test_2)
