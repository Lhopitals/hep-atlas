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
    # nombre de la señal a la cual se le mostrará el número de eventos
    name_signal = df_all.index.get_level_values('df_name').unique()[0]

    # se aplican todos los cortes
    for variable in cuts:
        # se grafica el numero de eventos
        numero_eventos_antes = df_all.query('df_name == @name_signal').shape[0]
        print(f'Numero eventos antes: {numero_eventos_antes}')
        
        #Definimos si el corte es un booleano. Se ocupa cuando se quieren cortar cosas del estilo Triggers
        if type(cuts[variable]) == type(True):
            print(f'Corte: {variable} == {cuts[variable]}')
            df_all = df_all[df_all[variable] == cuts[variable]]

            
        #Definimos si el corte es una lista, esto se ocupa para cuando se quieren cortar máximos y mínimos.   
        elif type(cuts[variable]) == type([]):
            corte_menor = cuts[variable][0]*scale[variable]
            corte_mayor = cuts[variable][1]*scale[variable]
            
            print(f'Corte: {variable} entre {cuts[variable]}')
            df_all = df_all[df_all[variable] < corte_mayor]
            df_all = df_all[df_all[variable] > corte_menor]

        
        #Definimos si el corte es un número entero. Se ocupa para cuando queremos separar los datos que tienen que ser un valor específico como un veto.
        elif type(cuts[variable]) == type(0):
            print(f'Corte: {variable} == {cuts[variable]}')
            df_all = df_all[df_all[variable] == cuts[variable]]


        # elif type(cuts[variable]) == type(''):
        #     print(f'Corte: {variable} == {cuts[variable]}')
        #     df_all = df_all[df_all[variable] == cuts[variable]]
            
        else:
            print("ADVERTENCIA: NO TOMA LA VARIABLE DEL CORTE")

        numero_eventos_despues = df_all.query('df_name == @name_signal').shape[0]
        print(f'Numero eventos antes: {numero_eventos_despues} \n')
             
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
    n_datos = [] # lista que va a guardar la cantidad de elementos de dataframe
    # n_datos.append(df.shape[0])

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
        n_datos.append(df_cut.shape[0])
        
    # Se devuelve un DataFrame con dos columnas
    df_eficiencias = pd.DataFrame({
        'n_datos': n_datos,
        'cortes': valores_cortes,
        'eficiencias': valores_eficiencias_variable
    })

    return df_eficiencias
    # return [valores_cortes, valores_eficiencias_variable]



def calc_eficiencias(df_all, variable):
    df_eficiencias = df_all.groupby(["origin", "df_name"]) \
                            .apply(lambda grupo: barrido_eficiencia_variable(grupo, variable))
    return df_eficiencias



def calc_bk_rejection(df_all, variable):
    df_eficiencias = calc_eficiencias(df_all, variable)
    df_eficiencias['bk_rejection'] = 1 - df_eficiencias['eficiencias']
    return df_eficiencias



def calc_bk_rejection_all_background(df_all, variable):
    df_background = df_all.query('origin == "background"')
    df_eficiencias = barrido_eficiencia_variable(df_background, variable)
    df_eficiencias['bk_rejection'] = 1 - df_eficiencias['eficiencias']
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
    plt.rcParams['font.size'] = 24 # estaba en 14
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "serif"
    plt.style.use('classic')


    ################# ELIJO LA FORMA DEL GRAFICO DEPENDIENDO LO QUE QUEREMOS GRAFICAR ################


    if ((graficar_eficiencia == True) and (graficar_significancia == True)):
        fig, axes = plt.subplots(3,1, figsize=(10,12), sharex=True, sharey=False,  gridspec_kw={'height_ratios': [2.5, 1.2, 1.2]})
        #fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        #fig.tight_layout(fig)
        
        ###################################### TO DO : ################################
        ###################### CORREGIR LOS BORDES DEL GRAFICO #######################
        
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

    best_cut_eficiencia = find_best_cut(df_all, variable, "eficiencia")
    best_cut_significancia = find_best_cut(df_all, variable, "significancia")


    ################## GRAFICO DE SIGNIFICANCIA ##########################


    if graficar_significancia == True:
        # calculo la significancia de la variable introducida
        cortes, significancia_variable = barrido_significancia_variable(df_all, variable)
        #Scatter de la significancia.
        scatter_significancia = sns.scatterplot(ax = eje_significancia, x = cortes, y = significancia_variable, marker=(8,2,0), color='coral', s=75) #Grafico pequeño
        scatter_significancia.set_xlabel(variable, fontdict={'size':12})
        scatter_significancia.set_ylabel('Significance', fontdict={'size':12})
        scatter_significancia.axvline(x = best_cut_significancia, color = 'red', label = 'corte significancia')
 

    # datos previos de los histogramas
    # color_palette = sns.color_palette("hls", len(backgrounds))
    # my_binwidth = (df_all.loc["signal"][variable].max() - df_all.loc["signal"][variable].min())/100.
    n_bins = 10

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
                             alpha=0.05,  
                             stat='density', 
                             common_norm=False, 
                             binrange=(df_all.loc["signal"][variable].min(), df_all.loc["signal"][variable].max()), 
                             binwidth = (df_all.loc["signal"][variable].max() - df_all.loc["signal"][variable].min())/n_bins,  
                             weights=calc_weight(df_all), 
                             element="step", 
                            #  fill=False,
                             )
    

    #se ponen labels y legends en el grafico
    histoplot.set_xlabel(str(variable), fontdict={'size':12})
    histoplot.set_ylabel('Normalised Events for ' + str(variable) , fontdict={'size':12})
    histoplot.ticklabel_format(style='plain', axis='y')
    histoplot.axvline(x = best_cut_significancia, color = 'red', label = 'corte significancia')
    histoplot.axvline(x = best_cut_eficiencia, color = 'blue', label = 'corte significancia')


    ################## GRAFICO DE EFICIENCIA ##########################
    # color_palette = sns.color_palette("hls", len(backgrounds))
    if graficar_eficiencia == True:

        ######################### EFICIENCIA ##########################
        
        eficiencias_signal = calc_eficiencias(df_all, variable).query('origin=="signal"')
        scatter_eficiencia = sns.scatterplot(ax = eje_eficiencia, 
                                             data=eficiencias_signal, 
                                             x = "cortes", 
                                             y = "eficiencias", 
                                             color = 'black', 
                                            #  hue=df_eficiencias.index.get_level_values('df_name'),
                                             #legend=True,
                                             marker=(8,2,0), 
                                             s=75)


        # modificaciones graficos eficiencias
        scatter_eficiencia.set_xlabel(variable, fontdict={'size':12})
        scatter_eficiencia.set_ylabel('Efficiency', fontdict={'size':12})
        
        ######################### REJECTION ##########################

        # calculo y grafico el background rejection de signal
        bk_rejection_background = calc_bk_rejection(df_all, variable).query('origin == "background"')
        sns.scatterplot(data=bk_rejection_background, 
                        x="cortes", 
                        y="bk_rejection", 
                        # color = 'black', 
                        hue=bk_rejection_background.index.get_level_values('df_name'),
                        #label = "bk rejection signal",
                        #legend=True, 
                        marker=(8,2,0), 
                        s=30
                        )

        # calculo el background rejection de todos los backgrounds unidos
        bk_rejection_all_background = calc_bk_rejection_all_background(df_all, variable)
        sns.scatterplot(data=bk_rejection_all_background, 
                        x="cortes", 
                        y="bk_rejection", 
                        color = 'black', 
                        # hue=bk_rejection_background.index.get_level_values('df_name'),
                        #label = "bk rejection signal",
                        #legend=True, 
                        marker=(8,2,0), 
                        s=30
                        )
        
        # grafico de la linea de corte
        scatter_eficiencia.axvline(x = best_cut_eficiencia, color = 'blue', label = 'corte eficiencia')
        
        ####################### BARRAS DE ERROR ##########################
        #Calculo la desviación estándar de la lista eficiencia y se agregan al gráfico de eficiencia.
        # std = np.std(eficiencia_variable)
        plt.errorbar(x = eficiencias_signal["cortes"], 
                     y = eficiencias_signal["eficiencias"], 
                     yerr = np.sqrt(1/eficiencias_signal["n_datos"]), # multiplicar por desviacion estandar?, tendría que guardarla en la funcion de las eficiencias
                     fmt='none', 
                     linestyle='none')
        #plt.xlim(0,1000)
    
    
    
    #plt.savefig('complete_Graph1_MET.eps', format = 'eps')
    #plt.savefig('complete_Graph1_MET.pdf', format = 'pdf')
    #plt.legend()
    plt.show()

    ############## VIOLIN PLOT ####################
    # sns.violinplot(data=df_all, x=variable, y=df_all.index.get_level_values('df_name'))
    # plt.show()

    ############## PIE PLOT ####################
    # df_all.groupby(level='origin').size().plot(kind='pie', autopct='%1.1f%%', startangle=90)
    # plt.show()

    ############## SIGNIFICANCIA A TRAVES DE LOS CORTES #################
    # sns.barplo()
    

################################################################################
############################### FIND BEST CUT ##################################
################################################################################


def find_best_cut(df_all, variable, method):
    if method == "significancia":
        cortes, significancia_variable = barrido_significancia_variable(df_all, variable)
        index_max_significance = significancia_variable.index(max(significancia_variable))
        maximo_corte = cortes[index_max_significance]
        return maximo_corte
        
    if method == "eficiencia":
        eficiencias_signal = calc_eficiencias(df_all, variable).query('origin=="signal"')["eficiencias"]

        bk_rejection = calc_bk_rejection_all_background(df_all, variable)
        bk_rejection_bk = bk_rejection["bk_rejection"]

        diferencia = abs(bk_rejection_bk.reset_index(drop=True)-eficiencias_signal.reset_index(drop=True))
        indice_minima_diferencia = diferencia.idxmin()
        corte_interseccion = bk_rejection["cortes"][indice_minima_diferencia]
        
        return corte_interseccion



################################################################################
################################# TESTING CUT ##################################
################################################################################

# se realizan los cortes superiores e inferiores a una lista de dataframes
def test_cuts(df_all, cuts, scale):

    df_original = df_all

    significancias = []
    eficiencias = []
    cortes = []
    variables = []
    n_datos = []
    n_datos_signal = []
    n_datos_background = []

    significancias.append(significance(df_all))
    eficiencias.append(efficiency(df_all, df_all))
    cortes.append(0)
    variables.append("")
    n_datos.append(df_all.shape[0])
    n_datos_signal.append(df_all.query('origin=="signal"').shape[0])
    n_datos_background.append(df_all.query('origin=="background"').shape[0])

    # se aplican todos los cortes
    for variable in cuts:
        
        #Definimos si el corte es un booleano. Se ocupa cuando se quieren cortar cosas del estilo Triggers
        if type(cuts[variable]) == type(True):
            print(f'Corte: {variable} == {cuts[variable]}')
            df_all = df_all[df_all[variable] == cuts[variable]]

            
        #Definimos si el corte es una lista, esto se ocupa para cuando se quieren cortar máximos y mínimos.   
        elif type(cuts[variable]) == type([]):
            corte_menor = cuts[variable][0]*scale[variable]
            corte_mayor = cuts[variable][1]*scale[variable]
            
            print(f'Corte: {variable} entre {cuts[variable]}')
            df_all = df_all[df_all[variable] < corte_mayor]
            df_all = df_all[df_all[variable] > corte_menor]

        
        #Definimos si el corte es un número entero. Se ocupa para cuando queremos separar los datos que tienen que ser un valor específico como un veto.
        elif type(cuts[variable]) == type(0):
            print(f'Corte: {variable} == {cuts[variable]}')
            df_all = df_all[df_all[variable] == cuts[variable]]


        elif type(cuts[variable]) == type(''):
            print(f'Corte: {variable} == {cuts[variable]}')
            df_all = df_all[df_all[variable] == cuts[variable]]
            
        else:
            print("ADVERTENCIA: NO TOMA LA VARIABLE DEL CORTE")
             
        significancias.append(significance(df_all))
        eficiencias.append(efficiency(df_original, df_all))
        cortes.append(df_all[variable])
        variables.append(variable)
        n_datos.append(df_all.shape[0])
        n_datos_signal.append(df_all.query('origin=="signal"').shape[0])
        n_datos_background.append(df_all.query('origin=="background"').shape[0])
    
    df_data = pd.DataFrame({
        'cortes': cortes,
        'variables': variables,
        # 'n_datos': n_datos,
        'n_datos_background': n_datos_background,
        'n_datos_signal': n_datos_signal,
        'eficiencias': eficiencias,
        'significancias': significancias
    })

    ####### STACKED ############
    # sns.barplot(data = df_data, x = "variables", y = "n_datos")
    # df_data.plot(x='variables', kind='bar', stacked=True,
    #     title='Stacked Bar Graph by dataframe')
    # axes = df_data.plot.bar(rot=0, subplots=True)
    # axes[1].legend(loc=2)  
    
    ax1 = plt.subplot(1,1,1)
    w = 0.3
    
    #plt.xticks(), will label the bars on x axis with the respective country names.
    x = np.arange(df_data['variables'].shape[0])
    plt.xticks(x + w /2, df_data['variables'], rotation='vertical')

    cmap_rojo =plt.get_cmap("Reds")

    valores_normalizados_signal = (df_data['n_datos_signal']) / (df_data['n_datos_background'].max())
    valores_normalizados_background = (df_data['n_datos_background']) / (df_data['n_datos_background'].max())

    cmap_rojo_signal = cmap_rojo(valores_normalizados_signal)
    cmap_rojo_background = cmap_rojo(valores_normalizados_background)

    background =ax1.bar(x, df_data['n_datos_background'], width=w, color=cmap_rojo_background, align='center')
    #The trick is to use two different axes that share the same x axis, we have used ax1.twinx() method.
    ax2 = ax1.twinx()
    #We have calculated GDP by dividing gdpPerCapita to population.
    signal =ax2.bar(x + w, df_data['n_datos_signal'], width=w,color=cmap_rojo_signal,align='center')
    
    # pongo los label de cada barra 
    bars = ax1.bar(x, eficiencias)
    ax1.bar_label(bars, label_type='edge')

    #Set the Y axis label as GDP.
    plt.ylabel('N datos')
    #To set the legend on the plot we have used plt.legend()
    plt.legend([background, signal],['background', 'signal'])
    #To show the plot finally we have used plt.show().
    plt.show()
