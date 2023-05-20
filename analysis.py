#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import sys
import yaml
from tqdm import tqdm # sirve para ver la linea de carga al cargar los archivos
# import awkward as ak
# import vector

path = '/home/gabbo/hep-atlas/data/' # moverlo a main

################################################################
####################### LEER Y CORTAR DATOS ####################
################################################################

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
    for data in tqdm(datasets): 
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
            df = df[df[variable] > corte_menor]

        list_df_with_cuts.append(df)

    return list_df_with_cuts

#Se le da el corte eficiente
def do_cuts_efficiency(old_list_all_df, cuts):
    efficiency_df = pd.DataFrame()
    list_df_with_cuts = []
    lista_eficiencias = []

    for df in old_list_all_df:
        lista_variables = []
        lista_valores_corte_mayor = []
        lista_valores_corte_menor = []
        lista_eficiencias = []

        old_df = df

        for variable in cuts:
            corte_menor = cuts[variable][0]
            corte_mayor = cuts[variable][1]
            
            lista_variables.append(variable)
            lista_valores_corte_mayor.append(corte_mayor)
            lista_valores_corte_menor.append(corte_menor)

            df = df[df[variable] < corte_mayor]
            df = df[df[variable] > corte_menor]

            lista_eficiencias.append(df[variable].size/old_df[variable].size)

        list_df_with_cuts.append(df)

        efficiency_df["variable"] = lista_variables
        efficiency_df["valor_corte_mayor"] = lista_valores_corte_mayor
        efficiency_df["valor_corte_menor"] = lista_valores_corte_menor
        efficiency_df["eficiencias"] = lista_eficiencias

        lista_eficiencias.append(efficiency_df)
        print(efficiency_df)

    return lista_eficiencias

################################################################################
######################### SIGNIFICANCIA Y CORTES ###############################
################################################################################

# SIGNIFICANCE DEFINITION
def significance(signal: float,background: float) -> float:
    return np.sqrt(2 * abs( (signal+background) * np.log(1 + (signal/background)) - signal))

df_calculo_significancia = pd.DataFrame()

#EFFICIENCY 
def efficiency(df, df_cut):
    efficiencies = {}
    len_vbf_events = len(df)    #eventos totales
    len_cut_events = len(df_cut)    #eventos con corte
    for variable in df.columns:
        # print(df[variable])
        efficiencies.update({variable: df_cut[variable].size/df[variable].size})    #hay que hacerlo con la señal solamente
    return efficiencies
#background rejection 1 - eff 
#hola po






















#TO DO 
#1  -   TRANSFORMAR UNIDADES
#2  -   HACER Y ENTENDER EL CALC_sIGNIFICANCE 
#3  -   GRAFICAR




################################################################################
################################### GRAFICAR ###################################
################################################################################