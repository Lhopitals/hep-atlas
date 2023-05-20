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

##############################Funciones que ocuparemos#################
# Leer archivos de data_.yaml
def read_data_yaml(data_yaml_file):
    with open(data_yaml_file) as f:
        data_yaml = yaml.load(f, Loader=yaml.FullLoader)
    return data_yaml

#Leer archivos root
def read_root_file(path, filename, tree_name):
    file = uproot.open(path + filename)
    tree = file[tree_name]
    return tree

# crea un dataframe con todos los datos
def read_datasets(datasets, variables):
    list_all_df = []
    for data in datasets:
        datos = read_root_file(path, data, "miniT")
        df_data = datos.arrays(variables, library="pd")
        list_all_df.append(df_data)
    return list_all_df

#Se le da el corte
def do_cuts(old_list_all_df, cuts):
    list_df_with_cuts = []

    for df in old_list_all_df:
        for variable in cuts:
            corte_menor = cuts[variable][0]
            corte_mayor = cuts[variable][1]

            df = df[df[variable] < corte_mayor]
            df = df[df[variable] > corte_menor] # tira nan
        list_df_with_cuts.append(df)

    return list_df_with_cuts

################################################################################
######################### SIGNIFICANCIA Y CORTES ###############################
################################################################################

# SIGNIFICANCE DEFINITION
def significance(signal: float,background: float) -> float:
    return np.sqrt(2 * abs( (signal+background) * np.log(1 + (signal/background)) - signal))

df_calculo_significancia = pd.DataFrame()


#TO DO 
#1  -   TRANSFORMAR UNIDADES
#2  -   HACER Y ENTENDER EL CALC_sIGNIFICANCE (HABLAR CON LOS CABROS)
#3  -   GRAFICAR