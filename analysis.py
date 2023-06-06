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



#Se define la función con la que se escalan las variables para transformar unidades.
def scale_df(df, scale):
    for variable in scale:
        df[variable] = df[variable]#*scale[variable]
    return df



# crea una lista con todos los datasets introducidos en datasets
def read_datasets(datasets, variables, scale, path):
    list_all_df = []
    # se leen los df's introducidos en datasets
    for data in tqdm(datasets): 
        datos = read_root_file(path, data, "miniT")
        df_data = datos.arrays(variables, library="pd")
        df_data = scale_df(df_data, scale)

        # guardo el nombre del dataset 
        nombre = data.split('.', 1)[0] # elimino lo de despues del punto
        nombre = nombre.split('/', 1)[1] # elimino lo de antes del punto
        df_data.columns.name = nombre # le doy el nombre al dataframe

        # se guarda el df en la lista
        list_all_df.append(df_data)
        
    return list_all_df



#Se le da el corte
# se realizan los cortes superiores e inferiores a una lista de dataframes
def do_cuts(old_list_all_df, cuts, scale):

    # lista de df's cortados
    list_df_with_cuts = []

    # se realiza el corte para cada df
    for df in old_list_all_df:
        # se realiza el corte para cada variable
        for variable in cuts:
            # se guarda el corte mayor e inferior, el corte debe estar en la misma escala que los datos
            corte_menor = cuts[variable][0]*scale[variable]
            corte_mayor = cuts[variable][1]*scale[variable]

            # se hace el corte mayor e inferior, 
            df = df[df[variable] < corte_mayor]
            df = df[df[variable] > corte_menor]

        # se guardan los df's cortados en una lista
        list_df_with_cuts.append(df)
        
    return list_df_with_cuts



################################################################################
######################### SIGNIFICANCIA Y CORTES ###############################
################################################################################



# SIGNIFICANCE DEFINITION
def significance(signal, backgrounds):
    """
    signal es un dataframe 
    background es una lista de df
    """
    # se calcula la significancia con la variable "intlumi"
    signal_weight = signal["intLumi"].sum()

    # se calcula el peso de todos los background
    backgrounds_weight = 0
    for df in backgrounds:
        background_weight = df["intLumi"].sum()
        backgrounds_weight += background_weight 
    
    # se calcula la significancia con la fórmula proporcionada
    return np.sqrt(2 * abs( (signal_weight + backgrounds_weight) * np.log(1 + (signal_weight/backgrounds_weight)) - signal_weight))



def barrido_significancia_variable(signal, backgrounds, variable, derecha = True):
    """ 
    signal es solo un dataframe
    background es una lista de dataframes que son los backgrounds
    derecha sirve para saber si se va a la derecha o a la izquierda en el barrido
    variable es la variable con la cual se calculará la significancia
    """
    n_cuts = 100 # numero_iteraciones_cortes
    valores_eficiencias_variable = [] # lista donde se guardan las eficiencias 
    valores_cortes = [] # lista donde se guardan los cortes realizados
    empty_data = False # si hay datos vacíos se para el calculo de significancias

    # elimino los valores extremos de signal 
    low_data = signal[variable].quantile(0.01)
    high_data  = signal[variable].quantile(0.99)
    signal = signal[(signal[variable]>low_data) & (signal[variable]<high_data)]

    # elimino los valores extremos de background ################ PREGUNTAR SI USO LOS MISMOS CORTES PARA SIGNAL Y BACKGROUND AQUI! ################
    for background in backgrounds:
        background = background[(background[variable]>low_data) & (background[variable]<high_data)]

    # se realiza el barrido de cortes, y se calcula la significancia para cada corte
    for i in range(n_cuts):

        # hago un corte a signal que va aumentando en cada iteracion
        iteration_cut = signal[variable].quantile(i/n_cuts) # LUEGO USAR UNA FUNCION MÁS GENERAL QUE .QUANTILE
        if derecha==True:
            signal = signal[signal[variable]>iteration_cut]
        else:
            signal = signal[signal[variable]<iteration_cut]

        # si me quedo sin datos en el signal paro la simulación
        if signal.shape[0] == 0:
            empty_data = True

        # aplico el corte para cada background de la lista que va aumentando en cada iteración
        backgrounds_with_cuts = []
        for background in backgrounds:
            if derecha == True:
                background = background[background[variable]>iteration_cut] # duda sobre usar iteration_cut de signal o background!
            else:
                background = background[background[variable]>iteration_cut]
            backgrounds_with_cuts.append(background)

            # si se queda sin elementos se detiene la simulación
            if background.shape[0] == 0:
                empty_data = True

        # si signal o background se queda sin elementos se detiene la simulación
        if empty_data == True:
            break

        # se calcula la significancia con los nuevos cortes
        significancia_i = significance(signal, backgrounds_with_cuts)

        # se guarda la significancia y su corte respectivo
        valores_eficiencias_variable.append(significancia_i)
        valores_cortes.append(iteration_cut)
        
    return valores_cortes, valores_eficiencias_variable



################################################################################
################################# GRAFICAR #####################################
################################################################################



def graficar(signal, backgrounds, variable):

    # configuraciones para el gráfico
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['font.size'] = 14
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "serif"
    plt.style.use('classic')
    fig, axes = plt.subplots(2,1, figsize=(10,12), gridspec_kw={'height_ratios': [2, 1]})
    
    ################## GRAFICO DE SIGNIFICANCIA ##########################

    # calculo la significancia de la variable introducida
    cortes, significancia_variable = barrido_significancia_variable(signal, backgrounds, variable)

    #Scatter de la significancia.
    scatter = sns.scatterplot(ax = axes[1], x = cortes, y = significancia_variable, marker=(8,2,0), color='coral', s=75) #Grafico pequeño
    scatter.set_xlabel(variable, fontdict={'size':12})
    scatter.set_ylabel('Significance', fontdict={'size':12})
    scatter.set(xlim=(0,None))
    scatter.set(ylim=(-1,None))

    ################## HISTOGRAMA DE LOS DATOS ##########################

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

    # elimino los datos muy extremos de signal
    low_data = signal[variable].quantile(0.01)
    high_data = signal[variable].quantile(0.98)
    signal = signal[(signal[variable]>low_data) & (signal[variable]<high_data)]
    # print(signal["MET"])

    # datos previos de los histogramas
    bins_fix = (0,20)
    color_palette = sns.color_palette("hls", len(backgrounds))
    
    
    # se realiza el gráfico de los 
    sns.histplot(ax=axes[0], data=signal, x=variable, alpha=0.7, bins=20, label=signal, stat='density', common_norm=False)
    for i, background in enumerate(backgrounds_variable.index.get_level_values("simulation").unique()):
        data = backgrounds_variable.xs(background, level="simulation")
        histoplot = sns.histplot(ax=axes[0], data=data, x=variable, alpha=0.7, bins=20, label=background, stat='density', common_norm=False)
    
    #se ponen labels y legends en el grafico
    histoplot.set_xlabel(variable, fontdict={'size':12})
    histoplot.set_ylabel('Events for ' + str(variable) , fontdict={'size':12})
    histoplot.legend()

    
    
    #histoplot.set(ylim=(None,500000))
    #plt.savefig('cuts_2_alpha_1.eps', format = 'eps')
    #plt.savefig('cuts_2_alpha_1.pdf', format = 'pdf')
    #plt.legend()
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