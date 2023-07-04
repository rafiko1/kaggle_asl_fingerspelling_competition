import tensorflow as tf
import numpy as np
from .preprocess_features import *
from .augmentations import *

ROWS_PER_FRAME = 543
PAD_FRAMES = -100.
PAD_PHRASE = 61

MAX_LEN_PHRASE = 43
MAX_LEN_FRAMES = 384

def decode_tfrec(record_bytes):
    features = tf.io.parse_single_example(record_bytes, {
        'coordinates': tf.io.FixedLenFeature([], tf.string),
        'sequence_id': tf.io.FixedLenFeature([], np.int64),
        'phrase': tf.io.FixedLenFeature([], tf.string),
    })
    out = {}
    out['coordinates']  = tf.reshape(tf.io.decode_raw(features['coordinates'], tf.float32), (-1,ROWS_PER_FRAME,3))
    out['phrase'] = features['phrase'] 
    out['sequence_id'] = features['sequence_id']
    return out

def filter_nans_tf(x):  # ref point = landmark points
    # Gets mask for all frames if there is all NaNs inside and returns only frames with >=1 values
    mask = tf.math.logical_not(tf.reduce_all(tf.math.is_nan(tf.gather(x, POINT_LANDMARKS ,axis=1)), axis=[-2,-1]))
    x = tf.boolean_mask(x, mask, axis=0)
    return x

def preprocess_tfrecord(x, augment):
    coord = x['coordinates']
    coord = filter_nans_tf(coord)
    
    if augment:
        coord = augment_fn(coord, max_len=MAX_LEN_FRAMES)
    
    coord = tf.ensure_shape(coord, (None,ROWS_PER_FRAME,3))
    coord = Preprocess(max_len=MAX_LEN_FRAMES)(coord)[0]
    coord = tf.cast(coord, tf.float32)

    phrase =  preprocess_phrase(x['phrase']) 
    inp_phrase, out_phrase = phrase[:-1], phrase[1:]
    return (coord, inp_phrase), out_phrase 

def get_tfrec_dataset(tfrecords, 
                      # max_len_frames,
                      # max_len_phrase,
                      batch_size=64, 
                      drop_remainder=False, 
                      augment=False, shuffle=False, 
                      repeat=False, 
                      sequences = [1975433633]):
    
    ds = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.AUTOTUNE, compression_type='GZIP')
    ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    ds = ds.filter(lambda x: tf.reduce_all(tf.not_equal(x['sequence_id'], sequences))) #TODO change to tf.equal avoid confusion train/val
    ds = ds.map(lambda x: preprocess_tfrecord(x, augment=augment), tf.data.AUTOTUNE)

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(shuffle)
        options = tf.data.Options()
        options.experimental_deterministic = (False)
        ds = ds.with_options(options)

    if batch_size:
        ds = ds.padded_batch(batch_size, 
                             padding_values=((PAD_FRAMES, PAD_PHRASE), PAD_PHRASE), 
                             padded_shapes=(([MAX_LEN_FRAMES, CHANNELS],[MAX_LEN_PHRASE]), [MAX_LEN_PHRASE]), 
                             drop_remainder=drop_remainder)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds