from analysis import *

if __name__ == '__main__':

    # Se crean variables con lo recuperado del archivo .yaml
    data_yaml = read_data_yaml('previous_data.yaml')

    # de data_yaml obtenemos los nombres de los datasets
    datasets = data_yaml['datasets'].values() #esto devuelve solo los valores de cada variable.
    
    # de data_yaml obtenemos los nombres de las variables
    variables = data_yaml['recover_branches']

    # de data_yaml obtenemos los nombres de los cortes y los valores de los cortes menores y mayores
    cuts = data_yaml['cuts']
    
    # obtenemos una lista de dataframes a partir de los archivos root, los cuales son especificados en datasets y variables
    list_all_df = read_datasets(datasets, variables)

    # hacemos los cortes
    list_all_df_cortes = do_cuts(list_all_df, cuts)

    #print(list_all_df[0].describe())
    #print(do_cuts(list_all_df, cuts)[0].describe())

    diccionario = do_cuts_efficiency(list_all_df, cuts)
    print(diccionario)
    
    