import numpy as np
import pandas as pd  
import os
import shutil
from rdkit import Chem
import random
from itertools import combinations
from keras.models import load_model
import openbabel
from biopandas.mol2 import PandasMol2
from rdkit.Chem import Fragments
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolDrawing, DrawingOptions,MolsToImage
from rdkit import RDLogger
import feature 
import shutil   
import tracemalloc
#from memory_profiler import profile
import time


def canonize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)

def canonize_ls(ls_smi):
    ls_mol = []
    for smi in ls_smi:
        try:
            mol = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)
            ls_mol.append(mol)
            ls_mol = sorted(set(ls_mol))
        except:
            continue
    return ls_mol


def sub_pair(main_smi,fun_mol,fun_num):
    mainmol = Chem.MolFromSmiles(main_smi)
    mainmol_num = mainmol.GetNumAtoms()
    ls_pair = []
    for j in range(mainmol_num):
        for k in range(mainmol_num,mainmol_num+fun_num):
            ls_pair.append((j,k))
    return ls_pair

def sub_att(mainmol,fun_mol,pair):
    ls_submol = []

    mainmol = Chem.MolFromSmiles(mainmol)
    for i in pair:
        combo = Chem.CombineMols(mainmol,fun_mol)
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(i[0],i[1],order=Chem.rdchem.BondType.SINGLE)
        back = edcombo.GetMol()
        back = Chem.MolToSmiles(back, isomericSmiles=True, canonical=True)
        ls_submol.append(back)
    return ls_submol



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--mainmol',
                    help='main molecule', default='mainmol.csv')
    parser.add_argument('-f', '--function',
                    help='functional group you want to add on molecules', default='func.csv')
    parser.add_argument('-o', '--output_path',
                    help='the output path', default='OUTPUT')
    parser.add_argument('-n', '--number',
                    help='how many functional group you want to add', default=2, type=int)


    args = vars(parser.parse_args())

    # disable RDKit logger
    RDLogger.DisableLog('rdApp.*')


    time_start=time.time()

    #============= create directory "output" ==========================

    output_path = args['output_path']


    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
        os.mkdir(output_path)
    else :
        os.mkdir(output_path)

    #============= reload model which had been trained to predict IE and EA ================
    model_IE = load_model("model/ECFP_num_IE.h5")
    model_EA = load_model("model/ECFP_num_EA.h5")
    
    #=====================================================================
    round = args['number']

    #============== preparation for main molecules ===================
    ls_main_smi = pd.read_csv(args['mainmol'])['smiles'].tolist()
    ls_main_name = pd.read_csv(args['mainmol'])['name'].tolist()

    ls_main_smi_name = []
    for i in range(len(ls_main_smi)):
        ls_main_smi_name.append((ls_main_smi[i], ls_main_name[i]))
    #=============================================================

    #============== preparation for functional group ===================
    # data = pd.read_csv("func.csv")
    # data = data.loc[data['name'].isin(["fr_piperdine", "fr_pyridine"])]
    # # data = data.loc[data['name'].isin(["methoxy","fr_furan"])]
    # ls_func_sma = data['smarts'].tolist()
    # ls_func_name = data['name'].tolist()

    ls_func_sma = pd.read_csv(args['func'])['smarts'].tolist()
    ls_func_name = pd.read_csv(args['func'])['name'].tolist()

    ls_func_name_mol_num = []
    for i in range(len(ls_func_sma)):
        ls_func_name_mol_num.append((ls_func_name[i],Chem.MolFromSmarts(ls_func_sma[i]),Chem.MolFromSmarts(ls_func_sma[i]).GetNumAtoms()))
    #==================================================================

    #============== start to add functional group on the main molecule ===================


    for mainmol in ls_main_smi_name:
        print("==================== Main Molecules",mainmol[1],"====================")

        for func in ls_func_name_mol_num:
            
            dict_ls_submol = {}
            dict_ls_submol_IE = {}
            dict_ls_submol_EA = {}
            ls_submol_all = []
            ls_submol_IE_all = []
            ls_submol_EA_all = [] 
         
            print("================func",func[0],"==============")
            print('#####========== 1st round ===========')
            pair = sub_pair(mainmol[0],func[1],func[2])
            ls_submol = sub_att(mainmol[0],func[1],pair)
            dict_ls_submol['sub_mol_1st'] = canonize_ls(ls_submol)
            #=========== fingerprinter transformation ===========
            molecules = feature.molecules(dict_ls_submol['sub_mol_1st'])
            fp_ECFP_num = molecules.ECFP_num()
            #=========== prediction =========================
            dict_ls_submol_IE['sub_mol_1st'] = model_IE.predict(fp_ECFP_num)
            dict_ls_submol_EA['sub_mol_1st'] = model_EA.predict(fp_ECFP_num)
            
            ####=========== later round ==============
            for r in range(round-1):
                print('============'+str(r+2)+'st round'+'================')

                ls_submol_later = []
                    
                for submol in dict_ls_submol['sub_mol_{}'.format(str(r+1)+"st")]:
                    try:
                        for func_ in ls_func_name_mol_num:
                            pair = sub_pair(submol,func_[1],func_[2])
                            ls_submol = sub_att(submol,func_[1],pair)
                            ls_submol_later.extend(canonize_ls(ls_submol))
                    except AttributeError as error:                  # some molecules cannot be read. But reason still need to verified
                        continue
        
                
                dict_ls_submol['sub_mol_{}'.format(str(r+2)+"st")] = ls_submol_later
                #=========== fingerprinter transformation ===========
                molecules = feature.molecules(dict_ls_submol['sub_mol_{}'.format(str(r+2)+"st")])
                fp_ECFP_num = molecules.ECFP_num()
                #=========== prediction =========================
                dict_ls_submol_IE['sub_mol_{}'.format(str(r+2)+"st")] = model_IE.predict(fp_ECFP_num)
                dict_ls_submol_EA['sub_mol_{}'.format(str(r+2)+"st")] = model_EA.predict(fp_ECFP_num)
            ####============ Output (combine all the submol together into a list)====================

        
            #print(dict_ls_submol)
            for key, value in dict_ls_submol.items():
                for i in value:
                    ls_submol_all.append(i)
            for key, value in dict_ls_submol_IE.items():
                for i in value:
                    ls_submol_IE_all.append(i[0])
            for key, value in dict_ls_submol_EA.items():
                for i in value:
                    ls_submol_EA_all.append(i[0])
       

            data = pd.DataFrame({'smiles':ls_submol_all, 'IE':ls_submol_IE_all, 'EA':ls_submol_EA_all})
            data.to_csv(output_path+'/'+mainmol[1]+'_'+func[0]+'.csv', index=False)
            
    time_end=time.time()
    print('time cost',time_end-time_start,'s')













