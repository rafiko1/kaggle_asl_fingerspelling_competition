import tensorflow as tf 

def SCCE_loss(labels, logits):
    mask = labels != CFG['pad_phrase']
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    loss = loss_object(labels, logits)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

def SCCE_loss_with_smoothing(labels, logits):
    mask = labels != CFG['pad_phrase']
    labels = tf.cast(labels, tf.int32)
    labels = tf.one_hot(labels, CFG['num_classes'], axis=2)
    
    loss = tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True, label_smoothing = CFG['loss']['label_smoothing'])
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

def CTC_loss(labels, logits):
    label_length = tf.reduce_sum(tf.cast(labels != CFG['pad_phrase'], tf.int32), axis=-1)
    logit_length = tf.ones(tf.shape(logits)[0], dtype=tf.int32) * tf.shape(logits)[1]
    loss = tf.nn.ctc_loss(labels=labels,
                          logits=logits,
                          label_length=label_length,
                          logit_length=logit_length,
                          blank_index=CFG['pad_phrase'],
                          logits_time_major=False)
    loss = tf.reduce_mean(loss)
    return loss

# def CTCLoss(labels, logits):
#     # Compute the training-time loss value
#     batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
#     input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
#     label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

#     input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
#     label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

#     loss = tf.keras.backend.ctc_batch_cost(labels, logits, input_length, label_length)
#     return loss

def get_loss(config):
    global CFG
    CFG = config
    
    if CFG['loss']['name'] == 'SCCE_loss':
        return SCCE_loss
    
    elif CFG['loss']['name'] == 'SCCE_loss_with_smoothing':
        return SCCE_loss_with_smoothing
    
    elif CFG['loss']['name'] == 'CTC_loss':
        return CTC_loss
    
def get_metric(config):
    if CFG['metric']['name'] == 'SCCE_loss':
        return SCCE_loss
    
    elif CFG['metric']['name'] == 'SCCE_loss_with_smoothing':
        return SCCE_loss_with_smoothing
    
    elif CFG['metric']['name'] == 'CTC_loss':
        return CTC_loss