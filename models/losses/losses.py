import tensorflow as tf 

def SCCE_loss(label, pred):
    mask = label != PAD_PHRASE

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)

    return loss

def SCCE_loss_with_smoothing(y_true, y_pred):
    mask = y_true != PAD_PHRASE
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, NUM_CLASSES, axis=2)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True, label_smoothing=LABEL_SMOOTHING)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

def get_loss(CFG):
    LABEL_SMOOTHING = CFG['loss']['label_smoothing']
    if CFG['loss']['name'] == 'SCCE_loss':
        loss = SCCE_loss
    elif CFG['loss']['name'] == 'SCCE_loss_with_smoothing':
        loss = SCCE_loss_with_smoothing
    return loss