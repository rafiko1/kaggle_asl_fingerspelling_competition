from ..schedulers.lr_scheduler import * 

class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, wd_ratio):
        self.step_counter = 0
        self.wd_ratio = wd_ratio
    
    def on_epoch_begin(self, epoch, logs=None):
        model.optimizer.weight_decay = model.optimizer.learning_rate * self.wd_ratio
        print(f'learning rate: {model.optimizer.learning_rate.numpy():.2e}, weight decay: {model.optimizer.weight_decay.numpy():.2e}')
        

def get_callbacks(CFG):
    OUT_DIR = CFG['output_dir']
    EXP = CFG['experiment']
    FOLD = CFG['fold']
    
    callbacks = []
    if 'logger_cb' in CFG['callbacks']:
        logger_cb = tf.keras.callbacks.CSVLogger(f'{OUT_DIR}/{EXP}-fold{FOLD}-logs.csv')
        callbacks.append(logger_cb)
    
    if 'ckpt_cb' in CFG['callbacks']:
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(f'{OUT_DIR}/{EXP}-fold{FOLD}-fold{FOLD}-best.h5', 
                                                     monitor='loss', verbose=0, save_best_only=True,
                                                     save_weights_only=True, mode='min', save_freq='epoch')
        callbacks.append(ckpt_cb)
    
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
        wd_callback = WeightDecayCallback(wd_ratio = CFG['scheduler']['wd_ratio'])
        
        callbacks.append(lr_callback)
        callbacks.append(wd_callback)
    return callbacks