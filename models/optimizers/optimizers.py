import tensorflow as tf
import tensorflow_addons as tfa
from ..schedulers.lr_scheduler import *

def get_optimizer(CFG):
    if CFG['optimizer']['name'] == 'adam':
#        opt = tf.keras.optimizers.Adam(clipvalue = CFG['optimizer']['clipvalue'], clipnorm = CFG['optimizer']['clipnorm']) # .legacy
         opt = tf.keras.optimizers.Adam()
         
    elif CFG['optimizer']['name'] == 'radam':
        if CFG['scheduler']['name'] == 'one_cycle':
            schedule, decay_schedule = get_one_cycle_scheduler(CFG)
            opt = tfa.optimizers.RectifiedAdam(learning_rate=schedule, weight_decay=decay_schedule, 
                                               sma_threshold=CFG['optimizer']['sma_threshold'])
            opt = tfa.optimizers.Lookahead(opt,sync_period=CFG['optimizer']['sync_period'])
        else:
            opt = tfa.optimizers.RectifiedAdam(sma_threshold=CFG['optimizer']['sma_threshold']) 
            opt = tfa.optimizers.Lookahead(opt,sync_period=CFG['optimizer']['sync_period'])
            
    return opt
        