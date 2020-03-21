from keras.models import load_model
import tensorflow as tf
#from tensorflow.python.framework import ops
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1
# config.gpu_options.allow_growth = True
import numpy as np
import pandas as pd
import feature
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from RNN_property_predictor import Model
from feature import molecules, check_in_char_set
import time
# tf.compat.v1.disable_v2_behavior()

def ECFP_num_prediction_batch(ls_smi, batch_size=1024, model_IE=None, model_EA=None):
       '''
       this function is to predict IE and EA from ECFP_num batch by batch, when the amount of data is massive.
       Warning : you will lose some molecules by using this mehtod. 
       In this stage, I separate IE and EA temporily, in the future I consider train a model which predict IE and EA simultaneous
       But, I am not sure whether doing so can lead to a better prediction. 

       Example: 

       model_IE = load_model("model_ECFP/ECFP_num_IE.h5")
       model_EA = load_model("model_ECFP/ECFP_num_EA.h5")
       ls_smi = pd.read_csv("OUTPUT")['smiles'].tolist()
       IE, EA = ECFP_num_prediction_batch(ls_smi, batch_size=1024, model_IE=model_IE, model_EA=model_EA)

       '''
       total_num = (len(ls_smi)//batch_size)*batch_size
       epochs = int(total_num/batch_size)
       start = 0 
       out_IE = []
       out_EA = [] 
       for epoch in range(epochs):
              fp_ECFP_num = molecules(ls_smi[start:start+batch_size]).ECFP_num()
              out_IE.append(model_IE.predict(fp_ECFP_num))
              out_EA.append(model_EA.predict(fp_ECFP_num))
              start += batch_size
       out_IE = np.concatenate(out_IE, axis=0)
       out_EA = np.concatenate(out_EA, axis=0)
    
       return out_IE, out_EA


def ECFP_num_prediction(ls_smi, model_IE=None, model_EA=None):
       '''
       this function is to predict IE and EA from ECFP_num for all the molecules at once.
       Example: 

       model_IE = load_model("model_ECFP/ECFP_num_IE.h5")
       model_EA = load_model("model_ECFP/ECFP_num_EA.h5")
       ls_smi = pd.read_csv("OUTPUT")['smiles'].tolist()
       IE, EA = ECFP_num_prediction(ls_smi, model_IE=model_IE, model_EA=model_EA)
       '''
       fp_ECFP_num = molecules(ls_smi).ECFP_num()
       return model_IE.predict(fp_ECFP_num), model_EA.predict(fp_ECFP_num)


def SMILES_onehot_prediction_batch(ls_smi, model_name=None, char_set=None, data_MP=None, batch_size=1024):
       '''
       the function is to predict IE and EA from SMILES one-hot encoding batch by batch, when the amount of data is massive. 
       It will return the prediction IE and EA simultaneously. 
       
       Example:
       
       char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
              "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
              "s", "O", "[", "Cl", "Br", "\\"]
       data_MP = pd.read_csv('MP_clean_canonize_cut.csv')
       ls_smi = pd.read_csv('OUTPUT.csv')['smiles'].tolist()
       ls_smi_new, IE, EA = SMILES_onehot_prediction_batch(ls_smi, model_name='model_SMILES/model',char_set=char_set, data_MP=data_MP, batch_size=2048)
       '''
       
       #========= normalization =========
       Y = np.asarray(data_MP.as_matrix()[:,1:], dtype=np.float32)  # 1.IE   2.EA 
       scaler_Y = StandardScaler()
       scaler_Y.fit(Y)
       #=================================
       total_num = (len(ls_smi)//batch_size)*batch_size
       epochs = int(total_num/batch_size)

       tf.reset_default_graph()
       model = Model(seqlen_x=43, dim_x=39, dim_y=2, char_set=char_set)
       
       out = []
       ls_smi_new = [] 

       with model.session:
              model.reload(model_name=model_name)
              start = 0
              for epoch in range(epochs):
                     X, Xs, ls_smi_new_batch = molecules(ls_smi[start:start+batch_size]).one_hot(char_set)
                     Y_hat = scaler_Y.inverse_transform(model.predict(X))
                     ls_smi_new.extend(ls_smi_new_batch)
                     out.append(Y_hat)
                     start += batch_size
              Y_hat = np.concatenate(out, axis=0)
       
       return ls_smi_new, Y_hat[:,0], Y_hat[:,1]


def SMILES_onehot_prediction(ls_smi, model_name=None, char_set=None, data_MP=None):
       '''
       the function is to predict IE and EA from SMILES one-hot encoding for all the molecules at once.
       It will return the prediction IE and EA simultaneously. 
       
       Example:
       
       char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
              "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
              "s", "O", "[", "Cl", "Br", "\\"]
       data_MP = pd.read_csv('MP_clean_canonize_cut.csv')
       ls_smi = pd.read_csv('OUTPUT.csv')['smiles'].tolist()
       ls_smi_new, IE, EA = SMILES_onehot_prediction(ls_smi, model_name=',char_set=char_set, data_MP=data_MP)
       '''
       
       #========= IE and EA normalization =========
       Y = np.asarray(data_MP.as_matrix()[:,1:], dtype=np.float32)  # 1.IE   2.EA 
       scaler_Y = StandardScaler()
       scaler_Y.fit(Y)
       #===========================================
       tf.reset_default_graph()
       model = Model(seqlen_x=43, dim_x=39, dim_y=2, char_set=char_set)

       with model.session:
              model.reload(model_name=model_name)
              X, Xs, ls_smi_new = molecules(ls_smi).one_hot(char_set)
              Y_hat = scaler_Y.inverse_transform(model.predict(X))
       
       return ls_smi_new, Y_hat[:,0], Y_hat[:,1]








        
if __name__ == '__main__':
       
       # start = time.time()
       
       # model_IE = load_model("model_ECFP/ECFP_num_IE.h5")
       # model_EA = load_model("model_ECFP/ECFP_num_EA.h5")
       # ls_smi = ['N#C[SH](N)(C=O)O1C=CN=C1','CCOOCOC(=O)OCC','CCOC(=O)OCF','CCOC(=O)OCOOC','COC(P)(C(F)(F)F)C(F)(F)F']
       # IE, EA = ECFP_num_prediction(ls_smi, model_IE=model_IE, model_EA=model_EA)

       # data = pd.DataFrame(columns=['smiles', 'IE', 'EA'])
       # data['smiles'] = ls_smi
       # data['IE'] = IE
       # data['EA'] = EA
       # data.to_csv("result.csv", index=False)
       
       # end = time.time()
       # print("the execution time "+str(end-start))



       start = time.time()

       char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
              "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
              "s", "O", "[", "Cl", "Br", "\\"]
       data_MP = pd.read_csv('MP_clean_canonize_cut.csv')
       ls_smi = ['N#C[SH](N)(C=O)O1C=CN=C1','CCOOCOC(=O)OCC','CCOC(=O)OCF','P(F)(F)(F)(F)(F)F.[Zn]','CCOC(=O)OCOOC','COC(P)(C(F)(F)F)C(F)(F)F']
       ls_smi_new, IE, EA = SMILES_onehot_prediction(ls_smi, model_name='model_SMILES/model',char_set=char_set, data_MP=data_MP)
       
       data = pd.DataFrame(columns=['smiles', 'IE', 'EA'])
       data['smiles'] = ls_smi_new
       data['IE'] = IE
       data['EA'] = EA
       data.to_csv("result.csv", index=False)
       
       end = time.time()
       print("the execution time "+str(end-start))


       





   





