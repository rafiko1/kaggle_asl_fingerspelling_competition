import tensorflow as tf
from tf_utils.schedules import OneCycleLR, ListedLR
import math
import matplotlib.pyplot as plt
import numpy as np

# From https://www.kaggle.com/code/markwijkhuizen/aslfr-transformer-training-inference#Transformer
def lrfn(current_step, num_warmup_steps, lr_max, num_training_steps, num_cycles=0.50, warmup_method = 'log'):
    if current_step < num_warmup_steps:
        if warmup_method == 'log':
            return float(lr_max) * 0.10 ** (float(num_warmup_steps) - float(current_step))
        else:
            return float(lr_max) * 2 ** -(float(num_warmup_steps) - float(current_step))
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr_max = float(lr_max)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max

def plot_lr_schedule(lr_schedule, epochs):
    fig = plt.figure(figsize=(20, 10))
    plt.plot([None] + lr_schedule + [None])
    # X Labels
    x = np.arange(1, epochs + 1)
    x_axis_labels = [i if epochs <= 40 or i % 5 == 0 or i == 1 else None for i in range(1, epochs + 1)]
    plt.xlim([1, epochs])
    plt.xticks(x, x_axis_labels) # set tick step to 1 and let x axis start at 1
    
    # Increase y-limit for better readability
    plt.ylim([0, max(lr_schedule) * 1.1])
    
    # Title
    schedule_info = f'start: {lr_schedule[0]:.1E}, max: {max(lr_schedule):.1E}, final: {lr_schedule[-1]:.1E}'
    plt.title(f'Step Learning Rate Schedule, {schedule_info}', size=18, pad=12)
    
    # Plot Learning Rates
    for x, val in enumerate(lr_schedule):
        if epochs <= 40 or x % 5 == 0 or x is epochs - 1:
            if x < len(lr_schedule) - 1:
                if lr_schedule[x - 1] < val:
                    ha = 'right'
                else:
                    ha = 'left'
            elif x == 0:
                ha = 'right'
            else:
                ha = 'left'
            plt.plot(x + 1, val, 'o', color='black');
            offset_y = (max(lr_schedule) - min(lr_schedule)) * 0.02
            plt.annotate(f'{val:.1E}', xy=(x + 1, val + offset_y), size=12, ha=ha)
    
    plt.xlabel('Epoch', size=16, labelpad=5)
    plt.ylabel('Learning Rate', size=16, labelpad=5)
    plt.grid()
    plt.show()
        
def get_one_cycle_scheduler(CFG):
    schedule = OneCycleLR(CFG['lr'], 
                          CFG['n_epochs'], 
                          warmup_epochs=CFG['n_epochs']*CFG['scheduler']['warmup'], 
                          steps_per_epoch=steps_per_epoch, 
                          decay_epochs=CFG['n_epochs'], 
                          lr_min=CFG['scheduler']['lr_min'], 
                          decay_type=CFG['scheduler']['decay_type'], 
                          warmup_type='linear')
    
    decay_schedule = OneCycleLR(CFG['lr']*CFG['weight_decay'], 
                                CFG['n_epochs'], 
                                warmup_epochs=CFG['n_epochs']*CFG['warmup'], 
                                steps_per_epoch=steps_per_epoch, 
                                decay_epochs=CFG['n_epochs'], 
                                lr_min=CFG['scheduler']['lr_min']*CFG['scheduler']['weight_decay'], 
                                decay_type=CFG['scheduler']['decay_type'], 
                                warmup_type='linear')
    
    return schedule, decay_schedule

def get_scheduler(CFG):
    pass
    # if CFG['scheduler']['name'] == '
    
    