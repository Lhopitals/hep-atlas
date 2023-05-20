#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import yaml
import awkward as ak
import vector

path = '/home/gabbo/hep-atlas/data/' 


datasets=['signal/frvz_vbf_500757.root',
        #   'background/qcd.root',
        #   'background/top.root',
        #   #'wjets_ewk.root', 
        #   'background/wjets_strong_sh227.root',
        #   'background/zjets_ewk.root',
        #   'background/zjets_strong_sh227.root', 
          #'diboson.root'
          ]

variables = ["njet30", 
             "mjj", 
             "detajj", 
             "dphijj",
             "min_dphi_jetmet", 
             "LJjet1_gapRatio", 
             "LJjet1_BIBtagger", 
             "MET", 
             "neleSignal", 
             "nmuSignal", 
             "metTrig",
             "scale1fb", 
             "intLumi",
             "hasBjet", 
             "nLJmus20", 
             "LJjet1_timing",
             "LJjet1_DPJtagger", 
             "LJjet1_jvt", 
             "nLJjets20",
            ]



valor_muy_alto = 100000
cuts_up = [["njet30" , 1.34],
            ["mjj", 234]]

cuts_down = [["njet30", 1.34],
            ["mjj", 234]]

cuts_definitive = [["njet30", 1.34],
                    ["mjj", 234]]



#idea
# 0.25 met 0.5   [["met", 0.25, 0.5], ["mjj", 0.5, "empty"]]
# mjj 0.5
# cambiar todo a una unica lkista con 3 c olumnas
# si no hay nada usar np.nan o np.inf

##############################Funciones que ocuparemos#################

#Leer archivos root
def read_root_file(path, filename, tree_name):
    file = uproot.open(path + filename)
    tree = file[tree_name]
    return tree

# crea un dataframe con todos los datos
def read_datasets(datasets, variables, cuts_up, cuts_down):
    df_all = pd.DataFrame()
    for data in datasets:
        datos = read_root_file(path, data, "miniT")
        df_data = datos.arrays(variables, library="pd") 

        # for cut in cuts_up:
        #     df_data = do_cut(df_data, cut[0], cut[1], up = True)
        
        # for cut in cuts_down:
        #     df_data = do_cut(df_data, cut[0], cut[1], up = False)

    print(df_data)

def do_cut(df, variable_corte, valor_corte, up = True):
    if up:
        return df[df[variable_corte] < valor_corte]
    if not up:
        return df[df[variable_corte] > valor_corte]

read_root_file(path, 'signal/frvz_vbf_500757.root', 'miniT')
read_datasets(datasets, variables, cuts_up, cuts_down)


# def read_data(datasets):
#     df = pd.DataFrame(columns=["filtro"])
#     for data in datasets:
#         file = uproot.open(data)
#         tree = file["miniT"]

#         #Hasta aqui abre los datos datasets.root y extrae las variables del miniT
#         temp_df = tree.arrays(variables, library="pd") 
#         #Los guarda en un df temporal
#         temp_df = pd.DataFrame(temp_df).reset_index()
#         if data == "mT2_frvz_vbf_500757.root":
#                         df.loc[0, "filtro"] = "VBFfilter"
#                         df.loc[1, "filtro"] = "Dphi_jetjet"
#                         df.loc[2, "filtro"] = "MINDPHIJETMET"
#                         df.loc[3, "filtro"] = "Lepton_veto"
#                         df.loc[4, "filtro"] = "Bjet_veto"
#                         df.loc[5, "filtro"] = "Muonic_LJ_veto"                 
#                         df.loc[6, "filtro"] = "MET_trigg"
#                         df.loc[7, "filtro"] = "MET_min"
#                         df.loc[8, "filtro"] = "LJ_Gapratio"
#                         df.loc[9, "filtro"] = "caloDPJ1BIB_tagger"
#                         df.loc[10, "filtro"] = "JVT"
#                         df.loc[11, "filtro"] = "DPJ tagger 1"



