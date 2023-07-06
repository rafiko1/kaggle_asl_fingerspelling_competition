import tensorflow as tf 
import numpy as np
import json

################## PARAMS MediaPipe Landmarks #######################
NOSE=[
    1,2,98,327
]
LNOSE = [98]
RNOSE = [327]
LIP = [ 0,
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513,505,503,501]
RPOSE = [512,504,502,500]

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]

LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE #+POSE

NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6*NUM_NODES

################## PARAMS phrase table #######################

def get_lookup_table(CHAR_PATH_JSON):
    
    with open(CHAR_PATH_JSON) as json_file:
        LABEL_DICT = json.load(json_file)

    LABEL_DICT['S'] = 59
    LABEL_DICT['E'] = 60
    LABEL_DICT['<PAD>'] = 61

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=list(LABEL_DICT.keys()),
            values=list(LABEL_DICT.values()),
        ),
        default_value=tf.constant(-1),
        name="label_table"
    )
    
    return table

####################################################################################################################
############ Functions MediaPipe Landmarks ################# 
def tf_nan_mean(x, axis=0, keepdims=False): # mean of X,Y,Z axis along the frames. replacing NaNs with zero to not account for them
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = tf_nan_mean(x, axis=axis,  keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = POINT_LANDMARKS

    def call(self, inputs):
        if tf.rank(inputs) == 3:
            x = inputs[None,...]
        else:
            x = inputs

        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1,2], keepdims=True) # NaN mean of nose
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5,x.dtype), mean) # if mean is NaN replace by 0.5
        x = tf.gather(x, self.point_landmarks, axis=2) #N,T,P,C  <- Gather the specific landmarks
        std = tf_nan_std(x, center=mean, axis=[1,2], keepdims=True)

        x = (x - mean)/std

        if self.max_len is not None:
            x = x[:,:self.max_len]
        length = tf.shape(x)[1]
        x = x[...,:2]

        dx = tf.cond(tf.shape(x)[1]>1,lambda:tf.pad(x[:,1:] - x[:,:-1], [[0,0],[0,1],[0,0],[0,0]]),lambda:tf.zeros_like(x)) # Lag1
        dx2 = tf.cond(tf.shape(x)[1]>2,lambda:tf.pad(x[:,2:] - x[:,:-2], [[0,0],[0,2],[0,0],[0,0]]),lambda:tf.zeros_like(x)) # Lag2

        x = tf.concat([
            tf.reshape(x, (-1,length,2*len(self.point_landmarks))),
            tf.reshape(dx, (-1,length,2*len(self.point_landmarks))),
            tf.reshape(dx2, (-1,length,2*len(self.point_landmarks))),
        ], axis = -1)

        x = tf.where(tf.math.is_nan(x),tf.constant(0.,x.dtype),x)

        return x

############ Functions phrase ################# 

def preprocess_phrase(phrase, table):
    phrase = 'S' + phrase + 'E'
    phrase = tf.strings.bytes_split(phrase)
    phrase = table.lookup(phrase)
    return phrase
