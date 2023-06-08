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
############################### SIGNIFICANCIA ##################################
################################################################################


# SIGNIFICANCE DEFINITION
def significance(signal, backgrounds):
    """
    signal es un dataframe 
    background es una lista de df
    """
    # se calcula la significancia con la variable "intlumi"
    signal_weight = (signal["intLumi"]*signal["scale1fb"]).sum()

    # se calcula el peso de todos los background
    backgrounds_weight = 0
    for df in backgrounds:
        background_weight = (df["intLumi"]*df["scale1fb"]).sum()
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

    # # elimino los valores extremos de signal 
    # low_data = signal[variable].quantile(porcentaje_bajo)
    # high_data  = signal[variable].quantile(porcentaje_alto) ###### cambiarlo, poner el mismo numero de abajo
    # signal = signal[(signal[variable]>low_data) & (signal[variable]<high_data)]

    # # elimino los valores extremos de background
    # for background in backgrounds:
    #     background = background[(background[variable]>low_data) & (background[variable]<high_data)]

    valor_minimo = signal[variable].min()
    valor_maximo = signal[variable].max()

    # se realiza el barrido de cortes, y se calcula la significancia para cada corte
    for i in range(n_cuts):
        # hago un corte a signal que va aumentando en cada iteracion
        iteration_cut = valor_minimo + i*(valor_maximo-valor_minimo)/n_cuts
        
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
            
        # se calcula la significancia con los nuevos cortes
        significancia_i = significance(signal, backgrounds_with_cuts)

        # se guarda la significancia y su corte respectivo
        valores_eficiencias_variable.append(significancia_i)
        valores_cortes.append(iteration_cut)
        
    return valores_cortes, valores_eficiencias_variable

################################################################################
################################ EFICIENCIA ####################################
################################################################################

def efficiency(df, df_cut):
    df_total = df_cut.shape[0]/df.shape[0]
    return df_total


def barrido_eficiencia_variable(df, variable, derecha = True):
    n_cuts = 100 # numero_iteraciones_cortes
    valores_eficiencias_variable = [] # lista donde se guardan las eficiencias 
    valores_cortes = [] # lista donde se guardan los cortes realizados
    empty_data = False


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
            empty_data = True

        # se calcula la significancia con los nuevos cortes
        eficiencia_i = efficiency(df, df_cut)

        # se guarda la significancia y su corte respectivo
        valores_eficiencias_variable.append(eficiencia_i)
        valores_cortes.append(iteration_cut)
        
    return valores_cortes, valores_eficiencias_variable



################################################################################
################################### WEIGHT #####################################
################################################################################



def calc_weight(df):
    df_weight = df["intLumi"]*df["scale1fb"]
    return df_weight



################################################################################
################################# GRAFICAR #####################################
################################################################################



def graficar_sin_weights(signal, backgrounds, variable):

    # configuraciones para el gráfico
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['font.size'] = 14
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "serif"
    plt.style.use('classic')
    fig, axes = plt.subplots(2,1, figsize=(10,12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True, sharey=False)
    
    porcentaje_bajo = 0
    porcentaje_alto = 1

    ################## MODIFICACION DATOS PARA GRAFICAR ##########################

    # obtengo solo la variable que me interesa de los backgrounds
    list_backgrounds_variable = []
    keys=[]
    for background in backgrounds:
        variable_background = background[variable]

        # elimino los valores muy grandes
        low_data = variable_background.quantile(porcentaje_bajo)
        high_data  = variable_background.quantile(porcentaje_alto)
        variable_background = variable_background[(variable_background>low_data) & (variable_background<high_data)]

        # guardo los datos de la variable del background
        list_backgrounds_variable.append(variable_background)

        # guardo el nombre del background
        keys.append(background.columns.name)

    # guardo todos los datos de la variable en una sola columna, pero con diferente indice
    backgrounds_variable = pd.concat(list_backgrounds_variable, axis=0, keys=keys, names=["simulation", "ID"])
    backgrounds_variable = pd.DataFrame(backgrounds_variable, columns=[variable])

    # elimino los datos muy extremos de signal
    low_data = signal[variable].quantile(porcentaje_bajo)
    high_data = signal[variable].quantile(porcentaje_alto)
    signal = signal[(signal[variable]>low_data) & (signal[variable]<high_data)]
    # print(signal["MET"])

    ################## GRAFICO DE SIGNIFICANCIA ##########################

    # calculo la significancia de la variable introducida
    cortes, significancia_variable = barrido_significancia_variable(signal, backgrounds, variable)

    #Scatter de la significancia.
    scatter = sns.scatterplot(ax = axes[1], x = cortes, y = significancia_variable, marker=(8,2,0), color='coral', s=75) #Grafico pequeño
    scatter.set_xlabel(variable, fontdict={'size':12})
    scatter.set_ylabel('Significance', fontdict={'size':12})
    # scatter.set(xlim=(0,None))
    scatter.set(ylim=(-0.1,5))



    # datos previos de los histogramas
    #color_palette = sns.color_palette("hls", len(backgrounds))
    n_bins = 20
    
    ################## HISTOGRAMA DE LOS DATOS ##########################
    sns.histplot(ax=axes[0], data=signal, x=variable, alpha=0.7, stat='density', common_norm=False, label='signal', binrange=(signal[variable].min(), signal[variable].max()), binwidth = 10)
    for background_name in backgrounds_variable.index.get_level_values("simulation").unique():
        data = backgrounds_variable.xs(background_name, level="simulation")
        histoplot = sns.histplot(ax=axes[0], data=data, x=variable, alpha=0.7, label=background_name, stat='density', common_norm=False, binrange=(0, 1000), binwidth = 10)
    
    # print(backgrounds_variable.xs(background_name, level="simulation"))
    #se ponen labels y legends en el grafico
    histoplot.set_xlabel(str(variable), fontdict={'size':12})
    histoplot.set_ylabel('Normalised Events for ' + str(variable) , fontdict={'size':12})
    histoplot.legend()

    
    
    histoplot.set(ylim=(None,0.07))
    #plt.savefig('cuts_funcionando_sig.eps', format = 'eps')
    #plt.savefig('cuts_funcionando_sig.pdf', format = 'pdf')
    #plt.legend()
    plt.show()




def graficar2(signal, backgrounds, variable, graficar_significancia = True, graficar_eficiencia = True, aplicar_weights = True):

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
    
    porcentaje_bajo = 0
    porcentaje_alto = 0.98

    ################## MODIFICACION DATOS PARA GRAFICAR Y CALCULO WEIGHTS ##########################

    # obtengo solo la variable que me interesa de los backgrounds
    list_backgrounds_variable = []
    dict_backgrounds_weight = {}
    keys=[]
    for background in backgrounds:
        variable_background = background[variable]

        # elimino los valores muy grandes
        low_data = variable_background.quantile(porcentaje_bajo)
        high_data  = variable_background.quantile(porcentaje_alto)
        variable_background = variable_background[(variable_background>low_data) & (variable_background<high_data)]

        # guardo los datos de la variable del background
        list_backgrounds_variable.append(variable_background)

        # guardo los pesos del dataframe
        background_weight = calc_weight(background)
        
        dict_backgrounds_weight[background.columns.name] = background_weight

        # guardo el nombre del background
        keys.append(background.columns.name)

    # guardo todos los datos de la variable en una sola columna, pero con diferente indice
    backgrounds_variable = pd.concat(list_backgrounds_variable, axis=0, keys=keys, names=["simulation", "ID"])
    backgrounds_variable = pd.DataFrame(backgrounds_variable, columns=[variable])

    # elimino los datos muy extremos de signal
    low_data = signal[variable].quantile(porcentaje_bajo)
    high_data = signal[variable].quantile(porcentaje_alto)
    signal = signal[(signal[variable]>low_data) & (signal[variable]<high_data)]
    # print(signal["MET"])


    ################## GRAFICO DE SIGNIFICANCIA ##########################
    
    if graficar_significancia == True:
        # calculo la significancia de la variable introducida
        cortes, significancia_variable = barrido_significancia_variable(signal, backgrounds, variable)
        #Scatter de la significancia.
        scatter_significancia = sns.scatterplot(ax = eje_significancia, x = cortes, y = significancia_variable, marker=(8,2,0), color='coral', s=75) #Grafico pequeño
        scatter_significancia.set_xlabel(variable, fontdict={'size':12})
        scatter_significancia.set_ylabel('Significance', fontdict={'size':12})
        # scatter.set(xlim=(0,None))
        #scatter_significancia.set(ylim=(-0.1,5))


    # datos previos de los histogramas
    color_palette = sns.color_palette("hls", len(backgrounds))
    n_bins = 50
    my_binwidth = 20

    ################## HISTOGRAMA DE LOS DATOS ##########################
    if aplicar_weights == True:
        weight_signal = calc_weight(signal)
    else:
        weight_signal = np.ones(signal.shape[0])

    sns.histplot(ax=eje_histograma, data=signal, x=variable, alpha=0.7, stat='density', common_norm=False, label='signal', binrange=(signal[variable].min(), signal[variable].max()), binwidth = my_binwidth, weights=weight_signal)
    for background_name in backgrounds_variable.index.get_level_values("simulation").unique():
        data = backgrounds_variable.xs(background_name, level="simulation")
        if aplicar_weights == True:
            weight_background = dict_backgrounds_weight[background_name]
        else:
            weight_background = np.ones(data.shape[0])
        histoplot = sns.histplot(ax=eje_histograma, data=data, x=variable, alpha=0.7,  label=background_name, stat='density', common_norm=False, binrange=(signal[variable].min(), signal[variable].max()), binwidth = my_binwidth, weights=weight_background)
    
    #se ponen labels y legends en el grafico
    histoplot.set_xlabel(str(variable), fontdict={'size':12})
    histoplot.set_ylabel('Normalised Events for ' + str(variable) , fontdict={'size':12})
    histoplot.legend()
    histoplot.ticklabel_format(style='plain', axis='y')

    ################## GRAFICO DE EFICIENCIA ##########################
    if graficar_eficiencia == True:
        ############################################################################
        ################ HACER CICLO FOR PARA GRAFICAR EFICIENCIA Y ################
        ################# SIGNIFICANCIA PARA TODOS LOS BACKGROUNDS #################
        ############################################################################
        
        # calculo la significancia de la variable introducida
        cortes, eficiencia_variable = barrido_eficiencia_variable(signal, variable)
        #Scatter de la significancia.
        scatter_eficiencia = sns.scatterplot(ax = eje_eficiencia, x = cortes, y = eficiencia_variable, marker=(8,2,0), color='red', s=75) #Grafico pequeño
        scatter_eficiencia.set_xlabel(variable, fontdict={'size':12})
        scatter_eficiencia.set_ylabel('Efficiency', fontdict={'size':12})
        # scatter.set(xlim=(0,None))
        # scatter_eficiencia.set(ylim=(0,None))
        lista_background_rejection = []
        for i in range(len(eficiencia_variable)):
            background_rejection = 1 - eficiencia_variable[i]
            lista_background_rejection.append(background_rejection)
        plt.scatter(cortes, lista_background_rejection)    
        
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

def find_best_cut(signal, backgrounds, variable, method):
    if method == "significancia":
        cortes, significancia_variable = barrido_significancia_variable(signal, backgrounds, variable)
        index_max_significance = significancia_variable.index(max(significancia_variable))
        maximo_corte = cortes[index_max_significance]
        return maximo_corte
        
    if method == "eficiencia":
        cortes, signal_eficiencias = barrido_eficiencia_variable(signal, variable, derecha = True)

        
        
        for background in backgrounds:
            cortes_background, background_eficiencias = barrido_eficiencia_variable(background, variable, derecha = True)
            pass

        
        pass
