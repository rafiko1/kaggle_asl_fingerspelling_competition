from ..schedulers.lr_scheduler import * 
import os
from swa.tfkeras import SWA
import tensorflow_addons as tfa

class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, wd_ratio, model):
        self.step_counter = 0
        self.wd_ratio = wd_ratio
        self.model = model
    
    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.weight_decay = self.model.optimizer.learning_rate * self.wd_ratio
        print(f'learning rate: {self.model.optimizer.learning_rate.numpy():.2e}, weight decay: {self.model.optimizer.weight_decay.numpy():.2e}')
        

def get_callbacks(CFG, model):
    FOLD = CFG['fold']
    OUT_DIR = CFG['output_dir'] + '/' + CFG['experiment']
    os.makedirs(OUT_DIR, exist_ok=True)
    
    callbacks = []
    average_opt = ''
    if 'logger_cb' in CFG['callbacks']:
        logger_cb = tf.keras.callbacks.CSVLogger(f'{OUT_DIR}/fold{FOLD}-logs.csv')
        callbacks.append(logger_cb)
    
    if 'ckpt_cb' in CFG['callbacks']:
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(f'{OUT_DIR}/fold{FOLD}-{{epoch:02d}}.h5', 
                                                     monitor='val_loss', verbose=0,
                                                     save_best_only=False,
                                                     save_weights_only=True, mode='min',
                                                     save_freq='epoch',
                                                     period = CFG['save_period'])
        callbacks.append(ckpt_cb)
    
    if 'average_cb' in CFG['callbacks']:
        averge_cb = tfa.callbacks.AverageModelCheckpoint(filepath=f'{OUT_DIR}/average-fold{FOLD}-{{epoch:02d}}.h5',
                                                           update_weights=True)
        callbacks.append(average_cb)
        average_opt = 'average'
        
    if 'swa1_cb' in CFG['callbacks']:
        swa_cb = tfa.callbacks.AverageModelCheckpoint(filepath=f'{OUT_DIR}/swa-fold{FOLD}-{{epoch:02d}}.h5',
                                                           update_weights=True)
        callbacks.append(swa_cb)
        average_opt = 'swa'
        
    if 'swa2_cb' in CFG['callbacks']:
        swa_cb = SWA(start_epoch=20, 
                     lr_schedule='manual', 
                     batch_size=CFG['train']['batch_size'], # needed when using batch norm
                     verbose=1)
        callbacks.append(swa_cb)
        
    if 'lr_cb' in CFG['callbacks']:
        lr_schedule = [lrfn(step, 
                            num_warmup_steps=CFG['scheduler']['warmup'], 
                            lr_max=CFG['scheduler']['lr_max'], 
                            num_training_steps = CFG['n_epochs'],
                            num_cycles=CFG['scheduler']['num_cycles']) 
                       for step in range(CFG['n_epochs'])]
        
        # Plot the scheduler 
        plot_lr_schedule(lr_schedule, epochs=CFG['n_epochs'])
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: lr_schedule[step], verbose=0)
        wd_callback = WeightDecayCallback(wd_ratio = CFG['scheduler']['wd_ratio'], model = model)
        
        callbacks.append(lr_callback)
        callbacks.append(wd_callback)
    return callbacks, average_opt