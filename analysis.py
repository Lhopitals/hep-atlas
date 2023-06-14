#!/usr/bin/python3
import numpy as np
import pandas as pd
# import pyarrow as pa
# ! pip install --pre pandas==2.0.2 # ponerlo en la terminal
# print(pd.__version__)
# pd.options.mode.dtype_backend = 'pyarrow'
# pd.options.mode.data_manager="pyarrow"

# pd.options.mode.dtype_backend="pyarrow"

import matplotlib.pyplot as plt
import uproot
import sys
import yaml
from tqdm import tqdm # sirve para ver la linea de carga al cargar los archivos
import pint
import seaborn as sns
import mplhep as hep
hep.style.use(hep.style.ATLAS)
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
        df[variable] = df[variable]*scale[variable]
    return df



# crea una lista con todos los datasets introducidos en datasets
def read_datasets(signals, backgrounds, variables, scale, path):
    df_all = pd.DataFrame()
    df_all_signal = pd.DataFrame()
    df_all_background = pd.DataFrame()

    # se leen los df's introducidos en datasets
    for data in tqdm(signals): 
        datos = read_root_file(path, data, "miniT")
        df_data = datos.arrays(variables, library="pd")
        df_data = scale_df(df_data, scale)

        # guardo el nombre del dataset 
        nombre = data.split('.', 1)[0] # elimino lo de despues del punto
        nombre = nombre.split('/', 1)[1] # elimino lo de antes del punto
        df_data.columns.name = nombre # le doy el nombre al dataframe

        # añado llaves para diferenciar los dataframes
        df_data["df_name"] = nombre
        df_data["origin"] = "signal"

        # se guarda el df en la lista
        df_all_signal = pd.concat([df_all_signal, df_data], axis=0)
    
    # se leen los df's introducidos en datasets
    for data in tqdm(backgrounds): 
        datos = read_root_file(path, data, "miniT")
        df_data = datos.arrays(variables, library="pd")
        df_data = scale_df(df_data, scale)

        # guardo el nombre del dataset 
        nombre = data.split('.', 1)[0] # elimino lo de despues del punto
        nombre = nombre.split('/', 1)[1] # elimino lo de antes del punto
        df_data.columns.name = nombre # le doy el nombre al dataframe

        # añado llaves para diferenciar los dataframes
        df_data["df_name"] = nombre
        df_data["origin"] = "background"

        # se guarda el df en la lista
        df_all_background = pd.concat([df_all_background, df_data], axis=0)
    
    df_all = pd.concat([df_all_signal, df_all_background], axis=0)
    df_all.set_index(['origin', 'df_name'], inplace=True)
    return df_all

#Se le da el corte
# se realizan los cortes superiores e inferiores a una lista de dataframes
def do_cuts(df_all, cuts, scale):

    for variable in cuts:
        corte_menor = cuts[variable][0]*scale[variable]
        corte_mayor = cuts[variable][1]*scale[variable]

        df_all = df_all[df_all[variable] < corte_mayor]
        df_all = df_all[df_all[variable] > corte_menor]
        
    return df_all

################################################################################
############################### SIGNIFICANCIA ##################################
################################################################################



# SIGNIFICANCE DEFINITION
def significance(df_all):

    # se calcula la significancia con la variable "intlumi"
    signal_weight = (df_all.loc['signal']["intLumi"]*df_all.loc['signal']["scale1fb"]).sum()
    backgrounds_weight = (df_all.loc['background']["intLumi"]*df_all.loc['background']["scale1fb"]).sum()

    # se calcula la significancia con la fórmula proporcionada
    return np.sqrt(2 * abs( (signal_weight + backgrounds_weight) * np.log(1 + (signal_weight/backgrounds_weight)) - signal_weight))



def barrido_significancia_variable(df_all, variable, derecha = True):
    n_cuts = 100 # numero_iteraciones_cortes
    valores_significancia_variable = [] # lista donde se guardan las eficiencias 
    valores_cortes = [] # lista donde se guardan los cortes realizados

    valor_minimo = df_all.loc["signal"][variable].min()
    valor_maximo = df_all.loc["signal"][variable].max()

    # se realiza el barrido de cortes, y se calcula la significancia para cada corte
    for i in range(n_cuts):
        # hago un corte a signal que va aumentando en cada iteracion
        iteration_cut = valor_minimo + i*(valor_maximo-valor_minimo)/n_cuts
        
        if derecha==True:
            df_all = df_all[df_all[variable]>iteration_cut]
        else:
            df_all = df_all[df_all[variable]<iteration_cut]
            
        # se calcula la significancia con los nuevos cortes
        significancia_i = significance(df_all)

        # se guarda la significancia y su corte respectivo
        valores_significancia_variable.append(significancia_i)
        valores_cortes.append(iteration_cut)
        
    return valores_cortes, valores_significancia_variable



################################################################################
################################ EFICIENCIA ####################################
################################################################################



def efficiency(df, df_cut):
    eficiencia = df_cut.shape[0]/df.shape[0]
    return eficiencia



def barrido_eficiencia_variable(df, variable, derecha = True):
    n_cuts = 100 # numero_iteraciones_cortes
    valores_eficiencias_variable = [] # lista donde se guardan las eficiencias 
    valores_cortes = [] # lista donde se guardan los cortes realizados


    valor_minimo = df[variable].min()
    valor_maximo = df[variable].max()

    # se realiza el barrido de cortes, y se calcula la significancia para cada corte
    for i in range(n_cuts):
        # hago un corte a signal que va aumentando en cada iteracion
        iteration_cut = valor_minimo + i*(valor_maximo-valor_minimo)/n_cuts
        
        if derecha==True:
            df_cut = df[df[variable]>iteration_cut]
        else:
            df_cut = df[df[variable]<iteration_cut]

        # si me quedo sin datos en el signal paro la simulación
        if df.shape[0] == 0:
            break

        # se calcula la significancia con los nuevos cortes
        eficiencia_i = efficiency(df, df_cut)

        # se guarda la significancia y su corte respectivo
        valores_eficiencias_variable.append(eficiencia_i)
        valores_cortes.append(iteration_cut)
        
    return valores_cortes, valores_eficiencias_variable



def calc_eficiencias(df_all, variable):
    df_eficiencias = pd.DataFrame()
    for df_name_i in df_all.index.get_level_values('df_name').unique():
        cortes, eficiencias = barrido_eficiencia_variable(df_all.query('df_name == @df_name_i'), variable)
        df_eficiencia = pd.DataFrame({'cortes':cortes, 'eficiencias':eficiencias})
        df_eficiencia["df_name"] = df_name_i
        df_eficiencias = pd.concat([df_eficiencias, df_eficiencia], axis=0)
    df_eficiencias.set_index(['df_name'], inplace=True)
    return df_eficiencias



################################################################################
################################### WEIGHT #####################################
################################################################################



def calc_weight(df):
    df_weight = df["intLumi"]*df["scale1fb"]
    return df_weight

# def aplicar_weight(df_all, variable):
#     df_all[variable] = df_all[variable]*calc_weight(df_all)
# PREGUNTAR SI SE APLICA A TO-DO O SOLO ES PARA GRAFICAR



################################################################################
################################# GRAFICAR #####################################
################################################################################



def graficar(df_all, variable, graficar_significancia = True, graficar_eficiencia = True, aplicar_weights = True):

    # configuraciones para el gráfico
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['font.size'] = 14
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "serif"
    plt.style.use('classic')


    ################# ELIJO LA FORMA DEL GRAFICO DEPENDIENDO LO QUE QUEREMOS GRAFICAR ################


    if ((graficar_eficiencia == True) and (graficar_significancia == True)):
        fig, axes = plt.subplots(3,1, figsize=(10,12), sharex=True, sharey=False)
        eje_histograma = axes[0]
        eje_significancia = axes[1]
        eje_eficiencia = axes[2]

    elif ((graficar_eficiencia == False) and (graficar_significancia == False)):
        fig, axes = plt.subplots(1,1, figsize=(10,12), sharex=True, sharey=False)
        eje_histograma = axes

    elif ((graficar_eficiencia == True) and (graficar_significancia == False)):
        fig, axes = plt.subplots(2,1, figsize=(10,12), sharex=True, sharey=False)
        eje_histograma = axes[0]
        eje_eficiencia = axes[1]
    
    else: # ((graficar_eficiencia == False) and (graficar_significancia == True))
        fig, axes = plt.subplots(2,1, figsize=(10,12), sharex=True, sharey=False)
        eje_histograma = axes[0]
        eje_significancia = axes[1]


    ################## MODIFICACION DATOS PARA GRAFICAR ##########################


    # pocentajes 
    porcentaje_bajo = 0.02
    porcentaje_alto = 0.98

    # elimino los valores extremos de df_all
    low_data = df_all.loc["signal"][variable].quantile(porcentaje_bajo)
    high_data  = df_all.loc["signal"][variable].quantile(porcentaje_alto) 
    df_all = df_all[(df_all[variable]>low_data) & (df_all[variable]<high_data)]


    ################## GRAFICO DE SIGNIFICANCIA ##########################


    if graficar_significancia == True:
        # calculo la significancia de la variable introducida
        cortes, significancia_variable = barrido_significancia_variable(df_all, variable)
        #Scatter de la significancia.
        scatter_significancia = sns.scatterplot(ax = eje_significancia, x = cortes, y = significancia_variable, marker=(8,2,0), color='coral', s=75) #Grafico pequeño
        scatter_significancia.set_xlabel(variable, fontdict={'size':12})
        scatter_significancia.set_ylabel('Significance', fontdict={'size':12})


    # datos previos de los histogramas
    # color_palette = sns.color_palette("hls", len(backgrounds))
    my_binwidth = 20

    ################## HISTOGRAMA DE LOS DATOS ##########################
    # if aplicar_weights == True:
    #     weight_signal = calc_weight(signal)
    # else:
    #     weight_signal = np.ones(signal.shape[0])


    histoplot = sns.histplot(ax=eje_histograma, 
                             data=df_all, 
                             x=variable, 
                             hue=df_all.index.get_level_values('df_name'),
                             legend=True,
                             alpha=0.7,  
                             stat='density', 
                             common_norm=False, 
                             binrange=(df_all.loc["signal"][variable].min(), df_all.loc["signal"][variable].max()), 
                             binwidth = my_binwidth, 
                             weights=calc_weight(df_all))
    

    #se ponen labels y legends en el grafico
    histoplot.set_xlabel(str(variable), fontdict={'size':12})
    histoplot.set_ylabel('Normalised Events for ' + str(variable) , fontdict={'size':12})
    histoplot.ticklabel_format(style='plain', axis='y')

    ################## GRAFICO DE EFICIENCIA ##########################
    # color_palette = sns.color_palette("hls", len(backgrounds))
    if graficar_eficiencia == True:
        
        df_eficiencias = calc_eficiencias(df_all, variable)
        scatter_eficiencia = sns.scatterplot(ax = eje_eficiencia, 
                                             data=df_eficiencias, 
                                             x = "cortes", 
                                             y = "eficiencias", 
                                             hue=df_eficiencias.index.get_level_values('df_name'),
                                             legend=True,
                                             marker=(8,2,0), 
                                             s=75)


        # modificaciones graficos eficiencias
        scatter_eficiencia.set_xlabel(variable, fontdict={'size':12})
        scatter_eficiencia.set_ylabel('Efficiency', fontdict={'size':12})

        ################## GRAFICO DE REJECTION ##########################
        # lista_background_rejection = []
        # for i in range(len(eficiencia_variable)):
        #     background_rejection = 1 - eficiencia_variable[i]
        #     lista_background_rejection.append(background_rejection)
        # plt.scatter(cortes, lista_background_rejection)    
        
        ################## GRAFICO DE DESVIACION ##########################
        #Calculo la desviación estándar de la lista eficiencia y se agregan al gráfico de eficiencia.
        # std = np.std(eficiencia_variable)
        # plt.errorbar(x = cortes, y = eficiencia_variable, yerr = std)
        #plt.xlim(0,1000)
    
    
    
    #plt.savefig('cuts_funcionando_sig_eff.eps', format = 'eps')
    #plt.savefig('cuts_funcionando_sig_eff.pdf', format = 'pdf')
    #plt.legend()
    plt.show()



################################################################################
############################### FIND BEST CUT ##################################
################################################################################


from scipy.interpolate import interp1d

def find_best_cut(df_all, variable, method):
    if method == "significancia":
        cortes, significancia_variable = barrido_significancia_variable(df_all, variable)
        index_max_significance = significancia_variable.index(max(significancia_variable))
        maximo_corte = cortes[index_max_significance]
        return maximo_corte
        
    if method == "eficiencia":
        df_all_eficiencias = calc_eficiencias(df_all, variable)
        print(df_all_eficiencias)

        # df_names = df_all.index.get_level_values('df_name').unique()
        # for df_name in df_all.index.get_level_values('df_name').unique():
        #     for i in range(len(df_names)):
        #         diferencia = df_all_eficiencias[df_name]-df_all_eficiencias[df_names[i]]
        #         pass

        df_names = df_all.index.get_level_values('df_name').unique()
        promedio_mejor_corte = 0
        numero_backgrounds = len(df_names)-1
        for i in range(numero_backgrounds):
            # calculo todas las distancias en el eje y entre  CON EL EJE Y YA SIRVE!!!!!
            diferencia = df_all_eficiencias.query('df_name == @df_names[0]')["eficiencias"] #-df_all_eficiencias.query('df_name == @df_names[i+1]')["eficiencias"]
            index_min_diferencia = diferencia.index(min(diferencia))
            # index_min_diferencia = diferencia.index(min(abs(diferencia)))
            maximo_corte = df_all_eficiencias[df_names[0]][index_min_diferencia]

            promedio_mejor_corte += maximo_corte/numero_backgrounds
        return promedio_mejor_corte