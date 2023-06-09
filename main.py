from analysis import *
import sys

if __name__ == '__main__':

    ########################################################
    ################## ELECCIÓN DE PATH ####################
    ########################################################

    # tomo la variable 
    input = int(sys.argv[1])

    # seleccionamos el path adecuado
    if input==1:
        path = '/home/gabbo/hep-atlas/data/'
        data_yaml = read_data_yaml('muonic_param.yaml')    #Parámetros del muonic

    elif input==2:
        path = '/home/tomilee/Desktop/Universidad/dark_photons/analisis_datos_github/hep-atlas/data/'
        data_yaml = read_data_yaml('calo_param_fix.yaml')      #Parámetros del calo

    else:
        print("ingrese un número válido")
        exit()

    ########################################################
    ########### LECTURA DATOS PREVIOS YAML #################
    ########################################################

    # Se crean variables con lo recuperado del archivo .yaml
    #data_yaml = read_data_yaml('previous_data_muon.yaml')  #De prueba para muonic
    #data_yaml = read_data_yaml('previous_data_calo.yaml')  #De prueba para muonic

    # data_yaml = read_data_yaml('muonic_param.yaml')    #Parámetros del muonic
    # data_yaml = read_data_yaml('calo_param.yaml')      #Parámetros del calo

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
    #print(df_all)

    ########################################################
    ####################### CORTES #########################
    ########################################################

    # pruebo como funcionan los cortes
    # test_cuts(df_all, cuts, scales)

    # hago los cortes definitivos
    df_all_cut = do_cuts(df_all, cuts, scales)

    ########################################################
    ####################### GRÁFICOS #######################
    ########################################################

    # ELIJO LA VARIABLE A GRAFICAR
    variable = "min_dphi_jetmet"

    graficar(df_all_cut, variable, derecha=False,
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

    # find_best_cut(df_all_cut, variable, "b")
    # print(find_best_cut(df_all_cut, "MET", "eficiencias"))