from analysis import *

if __name__ == '__main__':

    ########################################################
    ################## ELECCION DE PATH ####################
    ########################################################

    # seleccionamos el path adecuado
    print("Hola, ingresa en que maquina estás funcionando")
    print("1: Gabbo")
    print("2: Tomilee")

    input = int(input())

    if input==1:
        path = '/home/gabbo/hep-atlas/data/'

    elif input==2:
        path = '/home/tomilee/Desktop/Universidad/dark_photons/analisis_datos_github/hep-atlas/data/'

    else:
        print("ingrese un número válido")
        exit()

    ########################################################
    ########### LECTURA DATOS PREVIOS YAML #################
    ########################################################

    # Se crean variables con lo recuperado del archivo .yaml
    data_yaml = read_data_yaml('previous_data.yaml')

    # de data_yaml obtenemos los nombres de los datasets
    #datasets = data_yaml['datasets'].values() #esto devuelve solo los valores de cada variable.
    signals = data_yaml['signals'].values() #esto devuelve solo los valores de cada variable.
    backgrounds = data_yaml['backgrounds'].values() #esto devuelve solo los valores de cada variable.
    
    # de data_yaml obtenemos los nombres de las variables
    variables = data_yaml['recover_branches']

    # de data_yaml obtenemos las escalas de las variables
    scales = data_yaml['scale_variable']
    
    # de data_yaml obtenemos los nombres de los cortes y los valores de los cortes menores y mayores
    cuts = data_yaml['cuts']
    
    ########################################################
    ############# LECTURA DATOS ROOT FILES #################
    ########################################################

    # obtenemos una lista de dataframes a partir de los archivos root, los cuales son especificados en datasets y variables
    # list_all_df = read_datasets(datasets, variables, scales)
    list_all_signals = read_datasets(signals, variables, scales, path)
    list_all_background = read_datasets(backgrounds, variables, scales, path)
    
    # # obtenemos una lista de dataframes a partir de los archivos root, los cuales son especificados en datasets y variables
    # list_all_df = read_datasets(datasets, variables)

    ########################################################
    ####################### CORTES #########################
    ########################################################

    # hacemos los cortes
    # list_all_df_cortes = do_cuts(list_all_df, cuts)
    list_cut_df_signals = do_cuts(list_all_signals, cuts, scales)
    list_cut_df_backgrounds = do_cuts(list_all_background, cuts, scales)

    print(list_cut_df_signals[0].describe())

    ########################################################
    #################### SIGNIFICANCIA #####################
    ########################################################

    #ACAAAAAAAAAAAA QUEDAMOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    #print(list_all_df[0].describe())
    #print(do_cuts(list_all_df, cuts)[0].describe())
    #diccionario = do_cuts_efficiency(list_all_df, cuts)
    # print(diccionario)

    # print(list_all_df[1]['intLumi'].unique()) # [44.3  58.45  1.   36.1 ] # [44.3   1.   58.45 36.1 ]
    
    