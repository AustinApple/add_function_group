__author__ = "Chih Ao Liao, Ming Hsiu Wu "
import numpy as np
import pandas as pd  
import os
import shutil
from rdkit import Chem
import random
from itertools import combinations
from keras.models import load_model
import openbabel
from rdkit.Chem import Fragments
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolDrawing, DrawingOptions,MolsToImage
from rdkit import RDLogger
from feature import molecules
import shutil   
import tracemalloc
#from memory_profiler import profile
import time
import argparse

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
                    help='how many functional groups you want to add', default=2, type=int)
    parser.add_argument('--multi',action='store_true',
                    help='add different kinds of functional groups')
    parser.add_argument('--singel',action='store_true',
                    help='add one kind of function groups')
    args = vars(parser.parse_args())
    
    # disable RDKit logger
    RDLogger.DisableLog('rdApp.*')
 

    time_start=time.time()

    #============= create directory "output" ==========================
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
        os.mkdir(output_path)
    else :
        os.mkdir(output_path)
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
    ls_func_sma = pd.read_csv(args['function'])['smarts'].tolist()[:2]
    ls_func_name = pd.read_csv(args['function'])['name'].tolist()[:2]

    ls_func_name_mol_num = []
    for i in range(len(ls_func_sma)):
        ls_func_name_mol_num.append((ls_func_name[i],Chem.MolFromSmarts(ls_func_sma[i]),Chem.MolFromSmarts(ls_func_sma[i]).GetNumAtoms()))
    #==================================================================

    #============== start to add functional group on the main molecule ===================
    for mainmol in ls_main_smi_name:
        print("==================== Main Molecules",mainmol[1],"====================")

        for func in ls_func_name_mol_num:
            
            dict_ls_submol = {}
            ls_submol_all = []
         
            print("================func",func[0],"==============")
            print('#####========== 1st round ===========')
            pair = sub_pair(mainmol[0],func[1],func[2])
            ls_submol = sub_att(mainmol[0],func[1],pair)
            dict_ls_submol['sub_mol_1st'] = canonize_ls(ls_submol)
            ####=========== later round ==============
            ####=========== add one kind of function groups ==============
            if args['single']:
                for r in range(round-1):
                    print('============'+str(r+2)+'st round'+'================')
                    ls_submol_later = []
                    for submol in dict_ls_submol['sub_mol_{}'.format(str(r+1)+"st")]:
                        try:
                            pair = sub_pair(submol,func[1],func[2])
                            ls_submol = sub_att(submol,func[1],pair)
                            ls_submol_later.extend(canonize_ls(ls_submol))
                        except: 
                            continue
                    dict_ls_submol['sub_mol_{}'.format(str(r+2)+"st")] = ls_submol_later
            ####=========== add different kinds of function groups ==============
            if args['multi']:
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
            ####============ Output (combine all the submol together into a list according to different funcitonal groups)====================

        
            #print(dict_ls_submol)
            for key, value in dict_ls_submol.items():
                for i in value:
                    ls_submol_all.append(i)
       

            data = pd.DataFrame({'smiles':ls_submol_all})
            data.to_csv(output_path+'/'+mainmol[1]+'_'+func[0]+'.csv', index=False)
    #========= concat all the generated molecules =====================
    sub_file = os.listdir(args['output_path'])
    ls_df_all = []
    i=0
    # to prevent memory leaks
    for batch in range(len(sub_file)):
        ls_df = [pd.read_csv(output_path+'/'+name) for name in sub_file[i:i+1]]
        ls_df_all.extend(ls_df)     
        i = i + 1
    
    df = pd.concat(ls_df_all, ignore_index=True)
    df = df.drop_duplicates("smiles").reset_index(drop=True)
    df.to_csv(output_path+'/all.csv', index=False)
   
    time_end=time.time()
    print('time cost',time_end-time_start,'s')














