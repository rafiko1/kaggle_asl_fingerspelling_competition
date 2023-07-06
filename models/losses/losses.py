import tensorflow as tf 

def SCCE_loss(y_true, y_pred):
    mask = y_true != CFG['pad_phrase']
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    loss = loss_object(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

def SCCE_loss_with_smoothing(y_true, y_pred):
    mask = y_true != CFG['pad_phrase']
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, CFG['num_classes'], axis=2)
    
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True, label_smoothing = CFG['loss']['label_smoothing'])
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

def CTCLoss(labels, logits):
    label_length = tf.reduce_sum(tf.cast(labels != pad_token_idx, tf.int32), axis=-1)
    logit_length = tf.ones(tf.shape(logits)[0], dtype=tf.int32) * tf.shape(logits)[1]
    return tf.nn.ctc_loss(
            labels=labels,
            logits=logits,
            label_length=label_length,
            logit_length=logit_length,
            blank_index=pad_token_idx,
            logits_time_major=False
        )

def get_loss(config):
    global CFG
    CFG = config
    
    if CFG['loss']['name'] == 'SCCE_loss':
        return SCCE_loss
    
    elif CFG['loss']['name'] == 'SCCE_loss_with_smoothing':
        return SCCE_loss_with_smoothing