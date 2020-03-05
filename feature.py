import pandas
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import openbabel
import numpy as np 
import os 
import openbabel
import argparse
from keras.models import load_model
import tensorflow
from tensorflow.keras import preprocessing

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


    def one_hot(self, char_set):
        '''
        this function is to convert the smile inton one-hot encoding. 
        Parameter char_set includes all the character in all the SMILES. 
        '''
    
        char_to_int = dict((c,i) for i,c in enumerate(char_set))
        list_seq=[]
        for s in self.ls_smiles:
            seq=[]                
            j=0
            while j<len(s):
                if j<len(s)-1 and s[j:j+2] in char_set:
                    seq.append(char_to_int[s[j:j+2]])
                    j=j+2
                elif s[j] in char_set:
                    seq.append(char_to_int[s[j]])
                    j=j+1
            list_seq.append(seq)
             
        list_seq = preprocessing.sequence.pad_sequences(list_seq, padding='post')
        
        one_hot = np.zeros((list_seq.shape[0], list_seq.shape[1]+4, len(char_set)), dtype=np.int32)

        for si, ss in enumerate(list_seq):
            for cj, cc in enumerate(ss):
                one_hot[si,cj+1,cc] = 1
        #spend a time to figure out why it need to separate the one_hot into two arrays
            one_hot[si,-1,0] = 1
            one_hot[si,-2,0] = 1
            one_hot[si,-3,0] = 1

        return one_hot[:,0:-1,:], one_hot[:,1:,:]

def check_in_char_set(ls_smiles, char_set):
    ls_smiles_new = ls_smiles.copy()
    for s in ls_smiles:
        j=0
        while j<len(s):
            if j<len(s)-1 and s[j:j+2] in char_set:
                j=j+2
            elif s[j] in char_set:
                j=j+1
            elif s[j:j+2] not in char_set and s[j] not in char_set:
                ls_smiles_new.remove(s)
                break  
    return ls_smiles_new        
        

# test
if __name__ == '__main__':
    # model_IE = load_model("model/ECFP_num_IE.h5")
    # model_EA = load_model("model/ECFP_num_EA.h5")
    # a = molecules(['N#C[SH](N)(C=O)O1C=CN=C1'])
    # fp_ECFP = a.ECFP_num()
    # IE = model_IE.predict(fp_ECFP)
    # EA = model_EA.predict(fp_ECFP)

    # print(IE, EA)

    char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
           "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
           "s", "O", "[", "Cl", "Br", "\\"]
    ls_smi = ['N#C[SH](N)(C=O)O1C=CN=C1', 'P(F)(F)(F)(F)(F)F.[Zn]', 'COC(=O)C=O','[Mg].[BH4].[BH4]']
    ls_smi = check_in_char_set(ls_smi, char_set)
    



