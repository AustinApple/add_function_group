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
from feature import molecules
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

# pre-defined parameters

char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
           "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
           "s", "O", "[", "Cl", "Br", "\\"]
#data_uri='./data/ZINC_310k.csv'
data_MP = pd.read_csv('MP_clean_canonize_cut.csv')
Y = np.asarray(data_MP.as_matrix()[:,1:], dtype=np.float32)  # 1.IE   2.EA 
scaler_Y = StandardScaler()
scaler_Y.fit(Y)

ls_smi = pd.read_csv('OUTPUT/all.csv')['smiles'].tolist()
X, Xs = molecules(ls_smi).one_hot(char_set)

seqlen_x = X.shape[1]
dim_x = X.shape[2]
dim_y = Y.shape[1]

model = Model(seqlen_x=seqlen_x, dim_x=dim_x, dim_y=dim_y, char_set=char_set)
with model.session:
    model.reload(model_name='model_SMILES/model')
    Y_hat=scaler_Y.inverse_transform(model.predict(X))
    # print(Y_hat)
    print(Y_hat[:,0])
    print(Y_hat[:,1])






