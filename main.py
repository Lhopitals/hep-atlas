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

    df_all = read_datasets(signals, backgrounds, variables, scales, path)
    print(df_all)

    ########################################################
    ####################### CORTES #########################
    ########################################################

    df_all_cut = do_cuts(df_all, cuts, scales)

    ########################################################
    ####################### GRAFICOS #######################
    ########################################################

    graficar(df_all_cut, "jet1_pt", 
             graficar_significancia = True, 
             graficar_eficiencia = True, 
             aplicar_weights = True)
    
    ##################### GRAFICAR ALL #####################

    # for variable in variables:
    #     graficar(df_all_cut, variable, 
    #             graficar_significancia = True, 
    #             graficar_eficiencia = True, 
    #             aplicar_weights = True)

    ################### FIND BEST CUT ######################

    find_best_cut(df_all_cut, "MET", "eficiencia")