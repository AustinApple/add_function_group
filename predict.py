from keras.models import load_model
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
import numpy as np
import pandas as pd
import feature
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from RNN_property_predictor import Model
from feature import molecules, check_in_char_set

def ECFP_num_prediction(input, model_IE, model_EA):
    '''
    this function is to predict IE and EA from ECFP_num 
    In this stage, I separate IE and EA temporily, in the future I consider train a model which predict IE and EA simultaneous
    But, I am not sure whether doing so can have a better prediction. 
    
    Example: 
    
    model_IE = load_model("model_ECFP/ECFP_num_IE.h5")
    model_EA = load_model("model_ECFP/ECFP_num_EA.h5")
    a = molecules(['COC(=O)C=O'])
    fp_ECFP = a.ECFP_num()
    IE, EA = ECFP_num_prediction(fp_ECFP, model_IE=model_IE, model_EA=model_EA)
    '''
    return model_IE.predict(input), model_EA.predict(input)

def SMILES_onehot_prediction(input, model_name, char_set, data_MP):
    '''
    the function is to predict IE and EA from SMILES one-hot encoding
    It will return the prediction IE and EA simultaneously. 
    
    Example:
    
    char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
           "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
           "s", "O", "[", "Cl", "Br", "\\"]
    data_MP = pd.read_csv('MP_clean_canonize_cut.csv')
    
    ls_smi = pd.read_csv('OUTPUT/all.csv')['smiles'].tolist()
    X, Xs = molecules(ls_smi).one_hot(char_set)
    
    IE, EA = SMILES_onehot_prediction(input=X, model='model_SMILES/model',char_set=char_set, data_MP=data_MP)
    '''
    
    # predefine parameters
    Y = np.asarray(data_MP.as_matrix()[:,1:], dtype=np.float32)  # 1.IE   2.EA 
    scaler_Y = StandardScaler()
    scaler_Y.fit(Y)
    
    seqlen_x = input.shape[1]
    dim_x = input.shape[2]
    dim_y = Y.shape[1]

    tf.reset_default_graph()
    model = Model(seqlen_x=seqlen_x, dim_x=dim_x, dim_y=dim_y, char_set=char_set)

    with model.session:
        model.reload(model_name=model_name)
        Y_hat = scaler_Y.inverse_transform(model.predict(input))
        return Y_hat[:,0], Y_hat[:,1]
        
if __name__ == '__main__':
    # this is a test 

    char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
           "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
           "s", "O", "[", "Cl", "Br", "\\"]

    model_IE = load_model("model_ECFP/ECFP_num_IE.h5")
    model_EA = load_model("model_ECFP/ECFP_num_EA.h5")
    ls_smi = pd.read_csv('OUTPUT/all.csv')['smiles'].tolist()
    

    data = pd.DataFrame(columns=['smiles','IE_ECFP','EA_ECFP','IE_SMILES','EA_SMILES'])

    ls_smi = check_in_char_set(ls_smi, char_set)
    a = molecules(ls_smi)

    data['smiles'] = ls_smi 
    
    fp_ECFP = a.ECFP_num()
    IE, EA = ECFP_num_prediction(fp_ECFP, model_IE=model_IE, model_EA=model_EA)
    data['IE_ECFP'] = IE
    data['EA_ECFP'] = EA

    

    data_MP = pd.read_csv('MP_clean_canonize_cut.csv')
    
    ls_smi = pd.read_csv('OUTPUT/all.csv')['smiles'].tolist()
    X, Xs = molecules(ls_smi).one_hot(char_set)
    
    IE, EA = SMILES_onehot_prediction(input=X, model_name='model_SMILES/model',char_set=char_set, data_MP=data_MP)
    data['IE_SMILES'] = IE
    data['EA_SMILES'] = EA 

    data.to_csv("result.csv", index=False)








   





