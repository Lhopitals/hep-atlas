#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import sys
import yaml
from tqdm import tqdm # sirve para ver la linea de carga al cargar los archivos
import pint
import seaborn as sns
# import awkward as ak
# import vector

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
def read_datasets(datasets, variables, scale, path):
    list_all_df = []
    for data in tqdm(datasets): 
        datos = read_root_file(path, data, "miniT")
        df_data = datos.arrays(variables, library="pd")
        df_data = scale_df(df_data, scale)

        # guardo el nombre del dataset 
        nombre = data.split('.', 1)[0] # elimino lo de despues del punto
        nombre = nombre.split('/', 1)[1] # elimino lo de antes del punto
        df_data.columns.name = nombre # le doy el nombre al dataframe

        list_all_df.append(df_data)
    return list_all_df

#Se define la función con la que se escalan las variables para transformar unidades.
def scale_df(df, scale):
    for variable in scale:
        df[variable] = df[variable]#*scale[variable]
    return df     #REVISAR TOM

#Se le da el corte
def do_cuts(old_list_all_df, cuts, scale):
    list_df_with_cuts = []

    for df in old_list_all_df:
        for variable in cuts:
            corte_menor = cuts[variable][0]*scale[variable]
            corte_mayor = cuts[variable][1]*scale[variable]

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
def significance(signal, backgrounds):
    """
    signal es un dataframe 
    background es una lista de df
    """
    signal_weight = signal["intLumi"].sum()
    backgrounds_weight = 0
    for df in backgrounds:
        background_weight = df["intLumi"].sum()
        backgrounds_weight += background_weight 
    return np.sqrt(2 * abs( (signal_weight + backgrounds_weight) * np.log(1 + (signal_weight/backgrounds_weight)) - signal_weight))


def barrido_significancia_variable(signal, backgrounds, variable, derecha = True):
    """ 
    signal es solo un dataframe
    background es una lista de dataframes que son los backgrounds
    derecha sirve para saber si se va a la derecha o a la izquierda en el barrido
    variable es la variable con la cual se calculará la significancia
    """
    n_cuts = 100 # numero_iteraciones_cortes
    valores_eficiencias_variable = []
    valores_cortes = []

    # elimino los valores extremos 
    low_data = signal[variable].quantile(0.01)
    high_data  = signal[variable].quantile(0.99)
    signal = signal[(signal[variable]>low_data) & (signal[variable]<high_data)]

    for background in backgrounds:
        background = background[(background[variable]>low_data) & (background[variable]<high_data)]

    # de momento solo voy a hacerlo 100 veces para usar la funcion .quantile y no dejar el código engorroso, buscar una funcion mejor!
    for i in range(n_cuts):

        # hago un corte que va aumentando en cada iteracion
        iteration_cut = signal[variable].quantile(i/n_cuts)
        if derecha==True:
            signal = signal[signal[variable]>iteration_cut]
        else:
            signal = signal[signal[variable]<iteration_cut]

        # aplico el corte para cada background de la lista y los guardo en una nueva lista
        backgrounds_with_cuts = []
        for background in backgrounds:
            if derecha == True:
                background = background[background[variable]>iteration_cut] # duda sobre usar iteration_cut de signal o background!
            else:
                background = background[background[variable]>iteration_cut]
            backgrounds_with_cuts.append(background)

        # calculo la significancia 
        significancia_i = significance(signal, backgrounds_with_cuts)

        valores_eficiencias_variable.append(significancia_i)
        valores_cortes.append(iteration_cut)
        
    return valores_cortes, valores_eficiencias_variable



################################################################################
################################# GRAFICAR #####################################
################################################################################

def graficar(signal, backgrounds, significance, variable):
    #Uso de Latex en los ejes e instrucciones para graficar.
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['font.size'] = 14
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "serif"
    plt.style.use('classic')
    fig, axes = plt.subplots(2,1, figsize=(10,12), gridspec_kw={'height_ratios': [2, 1]})
    
    # calculo la significancia 
    cortes, significancia_variable = barrido_significancia_variable(signal, backgrounds, variable)

    #Scatter de la significancia.
    scatter = sns.scatterplot(ax = axes[1], x = cortes, y = significancia_variable, marker=(8,2,0), color='coral', s=75) #Grafico pequeño
    scatter.set_xlabel(variable, fontdict={'size':12})
    scatter.set_ylabel('Significance', fontdict={'size':12})

    # obtengo solo la variable que me interesa de los backgrounds
    list_backgrounds_variable = []
    keys=[]
    for background in backgrounds:
        variable_background = background[variable]

        # elimino los valores muy grandes
        low_data = variable_background.quantile(0.01)
        high_data  = variable_background.quantile(0.98)
        variable_background = variable_background[(variable_background>low_data) & (variable_background<high_data)]

        # guardo los datos de la variable del background
        list_backgrounds_variable.append(variable_background)

        # guardo el nombre del background
        keys.append(background.columns.name)

    # guardo todos los datos de la variable en una sola columna, pero con diferente indice
    backgrounds_variable = pd.concat(list_backgrounds_variable, axis=0, keys=keys, names=["simulation", "ID"])
    backgrounds_variable = pd.DataFrame(backgrounds_variable, columns=[variable])
    # print(backgrounds_variable)

    # elimino los datos muy extremos de signal
    low_data = signal[variable].quantile(0.01)
    high_data  = signal[variable].quantile(0.98)
    signal = signal[(signal[variable]>low_data) & (signal[variable]<high_data)]


    # se grafican los histoplots
    bins_fix = (0,20)
    sns.histplot(ax = axes[0], data=backgrounds_variable,x=variable, alpha=.7, bins=20, hue='simulation')
    histoplot = sns.histplot(ax = axes[0], data=signal,x=variable, alpha=.7, bins=bins_fix,legend=True)
    histoplot.set_xlabel(variable, fontdict={'size':12})
    histoplot.set_ylabel('Events for ' + str(variable) , fontdict={'size':12})
    histoplot.legend()
    #plt.savefig('fig_significance.eps', format = 'eps')
    #plt.savefig('fig_significance.pdf', format = 'pdf')
    plt.show()

################################################################################
################################ EFICIENCIA ####################################
################################################################################

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






















#TO DO 
#1  -   TRANSFORMAR UNIDADES
#2  -   HACER Y ENTENDER EL CALC_sIGNIFICANCE 
#3  -   GRAFICAR




################################################################################
################################### GRAFICAR ###################################
################################################################################