from __future__ import print_function
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True


import numpy as np
import pandas as pd
from preprocessing import smiles_to_seq, vectorize
import property_predictor

from preprocessing import get_property, canonocalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib

import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw

from matplotlib.pyplot import *
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

# pre-defined parameters
beta=10000.
char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
           "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
           "s", "O", "[", "Cl", "Br", "\\"]
#data_uri='./data/ZINC_310k.csv'
data_MP = './MP_new_canonize_cut.csv'


save_uri='./zinc_MP_PP.ckpt'


frac_val=0.1      # fraction for valuation 
ntst=2000          #  number of data for test in MP


test_times = 10 
log_test = np.zeros([10,2])



for random_seed in [9,9,9,9,9,9,9,9,9,9]:
    data = pd.read_csv(data_MP)
    np.random.seed(random_seed)
    random_id = np.random.permutation(data.shape[0])
    data = data.iloc[random_id,:]


    smiles_MP = data.as_matrix()[:,0] #0: SMILES

        

    end_index_MP = smiles_MP.shape[0]   # before this number is MP, after is zinc 
    # data preparation
    print('::: data preparation')


    Y = np.asarray(data.as_matrix()[:,1:], dtype=np.float32)  # 1.IE   2.EA 

    list_seq = smiles_to_seq(smiles_MP, char_set)


    Xs, X=vectorize(list_seq, char_set)
    print(X.shape)

#     tstX=X[end_index_MP-ntst:end_index_MP]
#     tstXs=Xs[end_index_MP-ntst:end_index_MP]
#     tstY=Y[end_index_MP-ntst:end_index_MP]


#     nL=int(len(Y)-ntst)       # subtract the number of test set 
#             # symbol the number of zinc data (unlabeled)  
#     nL_trn=int(nL*(1-frac_val))  
#     nL_val=int(nL*frac_val)


#     perm_id_nL=np.random.permutation(nL)
#     perm_id_nU=np.random.permutation([i for i in range(end_index_MP,len(X))])
#     print(perm_id_nU[:10])

#     trnX_L=X[perm_id_nL[:nL_trn]]
#     trnXs_L=Xs[perm_id_nL[:nL_trn]]
#     trnY_L=Y[perm_id_nL[:nL_trn]]

#     valX_L=X[perm_id_nL[nL_trn:nL_trn+nL_val]]
#     valXs_L=Xs[perm_id_nL[nL_trn:nL_trn+nL_val]]
#     valY_L=Y[perm_id_nL[nL_trn:nL_trn+nL_val]]


#     scaler_Y = StandardScaler()
#     scaler_Y.fit(Y)
#     trnY_L=scaler_Y.transform(trnY_L)
#     valY_L=scaler_Y.transform(valY_L)

#     ## model training
#     print('::: model training')

#     seqlen_x = X.shape[1]
#     dim_x = X.shape[2]
#     dim_y = Y.shape[1]
#     dim_z = 100
#     dim_h = 250

#     n_hidden = 3
#     batch_size = 200


#     model = property_predictor.Model(seqlen_x = seqlen_x, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, dim_h = dim_h,
#                         n_hidden = n_hidden, batch_size = batch_size, beta = float(beta), char_set = char_set)




#     with model.session:

#         model.reload(model_name='zinc_MP_PP.ckpt')

#         tstY_hat=scaler_Y.inverse_transform(model.predict(tstX))

#         for j in range(dim_y):
#             log_test[random_seed,j] = mean_absolute_error(tstY[:,j], tstY_hat[:,j])
#             print([j, mean_absolute_error(tstY[:,j], tstY_hat[:,j])])


#     tf.reset_default_graph()
       

# print("################# SUM UP  ################## ")
# print("the mean of IE tests is", str(np.mean(log_test, axis=0)[0]))
# print("the std of IE tests is", str(np.std(log_test, axis=0)[0]))

# print("the mean of EA tests is", str(np.mean(log_test, axis=0)[1]))
# print("the std of EA tests is", str(np.std(log_test, axis=0)[1]))





    # unconditional generation
    # for t in range(1000):
    #     smi = model.sampling_unconditional()
    #     list_smi_unconditional.append(smi)
    
    # data_unconditional = pd.DataFrame(list_smi_unconditional, columns=['SMILES']).to_csv("unconditional_smi.csv", index=False)

    
    ## conditional generation
    # yid = 0    # 1.IE   2.EA 
    # ytarget = 20.0
    # ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])
    
    # print('this is for conditional sampling')

    # for t in range(1000):

    #     smi = model.sampling_conditional(yid, ytarget_transform)
    #     list_smi_conditional.append(smi)
    # data_conditional = pd.DataFrame(list_smi_conditional, columns=['SMILES']).to_csv("conditional_smi_IE_20.csv", index=False)
    
# # start to featurize the smiles and prediction  for unconditional smiles
# print("i am here")
# os.system("python featurize.py -i unconditional_smi.csv -o unconditional_feature")
# data_unconditional = pd.read_csv("unconditional_smi.csv")

# ### this is for ECFP_NUM

# data_ECFP_NUM = pd.read_csv("unconditional_feature_ECFP_num.csv").drop(['SMILES'], axis=1)

# ECFPNUM_IE_predictor= joblib.load('./analyze_result/ECFPNUM_IE.pkl')
# data_unconditional["IE_ECFP_NUM"] = ECFPNUM_IE_predictor.predict(data_ECFP_NUM)

# ### this is for ECFP_SYBYL 
# ECFP = pd.read_csv("unconditional_feature_ECFP.csv")
# SYBYL = pd.read_csv("unconditional_feature_SYBYL.csv")
# ECFP_SYBYL = pd.concat([ECFP,SYBYL], axis=1)
# data_ECFP_SYBYL = ECFP_SYBYL.loc[:,~ECFP_SYBYL.columns.duplicated()].drop(['SMILES'], axis=1)

# SYBYL_ECFP_IE_predictor = joblib.load('./SYBYLECFP_IE.pkl')
# data_unconditional["IE_ECFP_SYBYL"] = SYBYL_ECFP_IE_predictor.predict(data_ECFP_SYBYL)

# # start to featurize the smiles and prediction  for conditional smiles


# os.system("python featurize.py -i conditional_smi.csv -o conditional_feature")

# data_unconditional = pd.read_csv("conditional_smi.csv")

# ### this is for ECFP_NUM

# data_ECFP_NUM = pd.read_csv("conditional_feature_ECFP_num.csv").drop(['SMILES'], axis=1)

# ECFPNUM_IE_predictor= joblib.load('./analyze_result/ECFPNUM_IE.pkl')
# data_conditional["IE_ECFP_NUM"] = ECFPNUM_IE_predictor.predict(data_ECFP_NUM)

# ### this is for ECFP_SYBYL 
# ECFP = pd.read_csv("conditional_feature_ECFP.csv")
# SYBYL = pd.read_csv("conditional_feature_SYBYL.csv")
# ECFP_SYBYL = pd.concat([ECFP,SYBYL], axis=1)
# data_ECFP_SYBYL = ECFP_SYBYL.loc[:,~ECFP_SYBYL.columns.duplicated()].drop(['SMILES'], axis=1)

# SYBYL_ECFP_IE_predictor = joblib.load('./SYBYLECFP_IE.pkl')
# data_conditional["IE_ECFP_SYBYL"] = SYBYL_ECFP_IE_predictor.predict(data_ECFP_SYBYL)

# # plot picture 
# fig, ax = subplots()
# data_IE = pd.read_csv('MP_new_canonize_cut.csv')
# data_IE['IE'].plot(kind='density')


# data_unconditional['IE_ECFP_SYBYL'].plot(kind='density')
# data_unconditional['IE_ECFP_NUM'].plot(kind='density')

# data_conditional['IE_ECFP_SYBYL'].plot(kind='density')
# data_conditional['IE_ECFP_NUM'].plot(kind='density')

# ax.legend(["origin", "unconditional_ECFP_SYBYL", "unconditional_ECFP_NUM", "conditional_ECFP_SYBYL", "conditional_ECFP_NUM"])
# ax.set_xlabel("IE (eV)",fontsize=15)
# ax.set_ylabel("Density",fontsize=15)
# ax.set_ylim(0,1)
# ax.set_xlim(0.0,10.0)
# ax.plot([5, 5], [0, 1], linestyle='--', dashes=(5, 5))
# fig.savefig("./distribution", transparent=True, dpi=500)