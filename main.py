from analysis import *
import sys

if __name__ == '__main__':

    ########################################################
    ################## ELECCION DE PATH ####################
    ########################################################

    # tomo la variable 
    input = int(sys.argv[1])

    # seleccionamos el path adecuado
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
    data_yaml = read_data_yaml('previous_data_muon.yaml')

    # de data_yaml obtenemos los nombres de los datasets
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
    list_all_signals = read_datasets(signals, variables, scales, path)
    list_all_background = read_datasets(backgrounds, variables, scales, path)
    
    ########################################################
    ####################### CORTES #########################
    ########################################################

    # hacemos los cortes
    list_cut_df_signals = do_cuts(list_all_signals, cuts, scales)
    list_cut_df_backgrounds = do_cuts(list_all_background, cuts, scales)

    ########################################################
    ####################### GRAFICOS #######################
    ########################################################

    #graficar_sin_weights(list_cut_df_signals[0],list_cut_df_backgrounds, 'MET')
    graficar2(list_cut_df_signals[0],list_cut_df_backgrounds, 'MET', 
              graficar_eficiencia=True, 
              graficar_significancia=True,
              aplicar_weights=False)

    graficar2(list_cut_df_signals[0],list_cut_df_backgrounds, 'MET', 
              graficar_eficiencia=True, 
              graficar_significancia=True,
              aplicar_weights=True)


    
    