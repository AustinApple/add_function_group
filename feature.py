import pandas
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import openbabel
import numpy as np 
import os 
from biopandas.mol2 import PandasMol2
import openbabel
import argparse
from keras.models import load_model
import tensorflow

class molecules():
    '''
    This class is to create a molecular object(SMILES)
    '''
    def __init__(self,ls_smiles):
        self.ls_smiles = ls_smiles

    
        
    def ECFP(self, radius=2, nbits=2048):
        '''
        convert a list of smiles into ECFP array
        '''
        arr_fp = np.zeros((len(self.ls_smiles), nbits))
        for i, smi in enumerate(self.ls_smiles):
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits).ToBitString()
            for j in range(nbits):
                arr_fp[i,j] = fp[j]
        return arr_fp
    
    def ECFP_num(self, radius=2, nbits=2048):
        '''
        convert a list of smiles into ECFP_num array
        '''
        arr_fp = np.zeros((len(self.ls_smiles), nbits), dtype=np.int8)
        for i, smi in enumerate(self.ls_smiles):
            try: 
                info={}
                mol = Chem.MolFromSmiles(smi)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, bitInfo=info).ToBitString()
                for bit in info:
                    arr_fp[i, bit] = len(info[bit])
            except:
                # print(smi+" has a problem when converted to ECFP !!")   need to verify what is going on.
                continue
        return arr_fp

# test

model = load_model("model/ECFP_num_IE.h5")
a = molecules(['C1COC(=O)O1', 'O=C(OCC)OCC'])
fp_ECFP = a.ECFP_num()
IE = model.predict(fp_ECFP)
