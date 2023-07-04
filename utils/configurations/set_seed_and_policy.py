import os 
import tensorflow as tf 
import gc
import tensorflow.keras.mixed_precision as mixed_precision
import random

# Seed all random number generators
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(seed=18)
tf.keras.backend.clear_session()
gc.collect()
tf.config.optimizer.set_jit(True)

mixed_precision.set_global_policy('mixed_bfloat16')