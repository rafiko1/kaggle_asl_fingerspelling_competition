import tensorflow as tf
import numpy as np
from .preprocess_features import *
from ..augmentations.augmentations import *

CHAR_PATH_JSON = 'asl_code/datasets/asl_fingerspelling/character_to_prediction_index.json'
table = get_lookup_table(CHAR_PATH_JSON) #global, not needed for inference

def decode_tfrec(record_bytes):
    features = tf.io.parse_single_example(record_bytes, {
        'coordinates': tf.io.FixedLenFeature([], tf.string),
        'sequence_id': tf.io.FixedLenFeature([], np.int64),
        'phrase': tf.io.FixedLenFeature([], tf.string),
    })
    out = {}
    out['coordinates']  = tf.reshape(tf.io.decode_raw(features['coordinates'], tf.float32), (-1, CFG['rows_per_frame'] ,3))
    out['phrase'] = features['phrase']
    out['sequence_id'] = features['sequence_id']
    out['N_frames'] = len(out['coordinates'])
    return out

def filter_nans_tf(x):  # ref point = landmark points
    # Gets mask for all frames if there is all NaNs inside and returns only frames with >=1 values
    mask = tf.math.logical_not(tf.reduce_all(tf.math.is_nan(tf.gather(x, POINT_LANDMARKS ,axis=1)), axis=[-2,-1]))
    x = tf.boolean_mask(x, mask, axis=0)
    return x

def preprocess_tfrecord(x, augment, return_inp_phrase=True):
    coord = x['coordinates']
    phrase = x['phrase']
    coord = filter_nans_tf(coord)
    
    if augment:
        if tf.random.uniform(())<CFG['augmentations']['reverse']:
            reverse=True
        else:
            reverse=False
        phrase =  preprocess_phrase(phrase, table, reverse=reverse)
        coord, phrase = augment_fn(coord, phrase, always=False, max_len=CFG['max_len_frames'], reverse=reverse)
    else:
        phrase =  preprocess_phrase(phrase, table, reverse=False)
    
    # coord = tf.ensure_shape(coord, (None,ROWS_PER_FRAME,3))
    coord = Preprocess(max_len=CFG['max_len_frames'])(coord)[0]
    coord = tf.cast(coord, tf.float32)
    
    inp_phrase, out_phrase = phrase[:-1], phrase[1:]
    if return_inp_phrase:
        return (coord, inp_phrase), out_phrase
    else:
        return (coord, out_phrase[:-1]), out_phrase[:-1] #coord, out_phrase[:-1]

def get_tfrec_dataset(tfrecords, 
                      config,
                      phase = 'train',
                      drop_sequences = [1975433633]):
    
    global CFG
    CFG = config
    
    ds = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.AUTOTUNE, compression_type='GZIP')
    ds = ds.map(lambda x: decode_tfrec(x), tf.data.AUTOTUNE)
    ds = ds.filter(lambda x: tf.reduce_all(tf.math.greater(x['N_frames'], CFG['min_number_frames'])))
    ds = ds.filter(lambda x: tf.reduce_all(tf.not_equal(x['sequence_id'], drop_sequences)))

    if CFG['model'] == 'encoder':
        ds = ds.map(lambda x: preprocess_tfrecord(x, CFG[phase]['augment'], return_inp_phrase=False), tf.data.AUTOTUNE)
    elif CFG['model'] == 'encoder-decoder':
        ds = ds.map(lambda x: preprocess_tfrecord(x, CFG[phase]['augment'], return_inp_phrase=True), tf.data.AUTOTUNE)
    
    if CFG[phase]['repeat']:
        ds = ds.repeat()

    shuffle = CFG[phase]['shuffle']
    if shuffle:
        ds = ds.shuffle(shuffle)
        options = tf.data.Options()
        options.experimental_deterministic = (False)
        ds = ds.with_options(options)
    
    if CFG['model'] == 'encoder':
        # ds = ds.padded_batch(CFG[phase]['batch_size'], 
        #              padding_values=(CFG['pad_frames'], CFG['pad_phrase']), 
        #              padded_shapes=([CFG['max_len_frames'], CHANNELS], [CFG['max_len_phrase']]), 
        #              drop_remainder=CFG[phase]['drop_remainder'])
        ds = ds.padded_batch(CFG[phase]['batch_size'], 
                             padding_values=((CFG['pad_frames'], CFG['pad_phrase']), CFG['pad_phrase']), 
                             padded_shapes=(([CFG['max_len_frames'], CHANNELS],CFG['max_len_phrase']), [CFG['max_len_phrase']]), 
                             drop_remainder=CFG[phase]['drop_remainder'])

    elif CFG['model'] == 'encoder-decoder':
        ds = ds.padded_batch(CFG[phase]['batch_size'], 
                             padding_values=((CFG['pad_frames'], CFG['pad_phrase']), CFG['pad_phrase']), 
                             padded_shapes=(([CFG['max_len_frames'], CHANNELS],CFG['max_len_phrase']), [CFG['max_len_phrase']]), 
                             drop_remainder=CFG[phase]['drop_remainder'])

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
