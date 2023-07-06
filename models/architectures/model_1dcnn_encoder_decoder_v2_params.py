import tensorflow as tf
from ...utils.preprocessing.preprocess_features import *

class ECA(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_average = tf.keras.layers.GlobalAveragePooling1D()
        self.conv = tf.keras.layers.Conv1D(1, 5, strides=1, padding="same", use_bias=False) #5?
    
    def call(self, inputs, mask=None):
        nn = self.global_average(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]
        return inputs * nn

class CausalDWConv1D(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        dilation_rate = 1
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(CFG['ksize']-1),0))
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
                            CFG['ksize'],
                            strides=1,
                            dilation_rate=dilation_rate,
                            padding='valid',
                            use_bias=False,
                            depthwise_initializer='glorot_uniform')

    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x

class Conv1DBlock(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(CFG['dim']*CFG['expand'], use_bias=True, activation=CFG['activation'])
        self.causal_dw_conv1d = CausalDWConv1D()
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=CFG['bn_momentum'])
        self.eca = ECA()
        self.dense2 = tf.keras.layers.Dense(CFG['dim'], use_bias=True)
        self.dropout = tf.keras.layers.Dropout(CFG['drop_rate'], noise_shape=(None,1,1))
        # self.add = tf.keras.layers.Add()
        
    def call(self, inp1):
        x = self.dense1(inp1)
        x = self.causal_dw_conv1d(x)
        x = self.batch_norm(x)
        x = self.eca(x)
        x = self.dense2(x)
        x = self.dropout(x)
        # x = self.add([x, inp1])
        return x
    
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], DTYPE)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_weights = tf.keras.layers.Softmax()(scaled_attention_logits, mask=mask)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output
    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):   # assert dim % self.num_heads == 0
        super().__init__()
        self.num_heads = CFG['num_heads']
        self.dim = CFG['dim']
        self.depth = CFG['dim'] // self.num_heads

        self.wq = tf.keras.layers.Dense(CFG['dim'])
        self.wk = tf.keras.layers.Dense(CFG['dim'])
        self.wv = tf.keras.layers.Dense(CFG['dim'])

        self.wo = tf.keras.layers.Dense(CFG['dim'])

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth). Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dim))  # (batch_size, seq_len_q, d_model)

        output = self.wo(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output

def positional_encoding(length):
    depth = CFG['dim']/2
    
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    
    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)
    
    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)
    
    return tf.cast(pos_encoding, dtype=DTYPE) # remove casting?

class PositionalEmbedding(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(CFG['num_classes'], CFG['dim'], mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(CFG['dim'], DTYPE)) # remove casting?
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

class TransformerBaseBlock(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.batch_norm1 = tf.keras.layers.BatchNormalization(momentum=CFG['bn_momentum'])
        # self.layer_norm1 = tf.keras.layers.LayerNormalization(dtype=tf.bfloat16)
        self.mha = MultiHeadAttention()
        self.dropout1 = tf.keras.layers.Dropout(CFG['drop_rate'], noise_shape=(None,1,1))
        self.add1 = tf.keras.layers.Add()

        self.batch_norm2 = tf.keras.layers.BatchNormalization(momentum=CFG['bn_momentum'])
        # self.layer_norm2 = tf.keras.layers.LayerNormalization(dtype=tf.bfloat16)

        self.dense1 = tf.keras.layers.Dense(CFG['dim']*CFG['expand'], use_bias=False, activation=CFG['activation'])
        self.dense2 = tf.keras.layers.Dense(CFG['dim'], use_bias=False)
        self.dropout2 = tf.keras.layers.Dropout(CFG['drop_rate'], noise_shape=(None,1,1))
        self.add2 = tf.keras.layers.Add()
        
    def call(self, input):
        x = self.batch_norm1(input)
        # x = self.layer_norm1(input)
        x = self.mha(x, x, x, mask=None)
        x = self.dropout1(x)
        x = self.add1([x, input])

        # Feed-forward 
        base_out = x
        x = self.batch_norm2(x)
        # x = self.layer_norm2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.add2([x, base_out])
        return x
    
def create_look_ahead_mask(size):
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = tf.cast(mask, DTYPE)
    return mask  # (seq_len, seq_len)
    
class TransformerCrossBlock(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.layer_norm = tf.keras.layers.LayerNormalization(dtype=tf.bfloat16)        
        self.mha1 = MultiHeadAttention()
        self.dropout1 = tf.keras.layers.Dropout(CFG['drop_rate'], noise_shape=(None,1,1))
        self.add1 = tf.keras.layers.Add()

        self.batch_norm1 = tf.keras.layers.BatchNormalization(momentum=CFG['bn_momentum'])
        self.batch_norm2 = tf.keras.layers.BatchNormalization(momentum=CFG['bn_momentum'])
        # self.layer_norm1 = tf.keras.layers.LayerNormalization(dtype=tf.bfloat16)        
        # self.layer_norm2 = tf.keras.layers.LayerNormalization(dtype=tf.bfloat16)
        self.mha2 = MultiHeadAttention()
        self.dropout2 = tf.keras.layers.Dropout(CFG['drop_rate'], noise_shape=(None,1,1))
        self.add2 = tf.keras.layers.Add()

        self.batch_norm3 = tf.keras.layers.BatchNormalization(momentum=CFG['bn_momentum'])
        # self.layer_norm3 = tf.keras.layers.LayerNormalization(dtype=tf.bfloat16)  

        self.dense1 = tf.keras.layers.Dense(CFG['dim']*CFG['expand'], use_bias=False, activation=CFG['activation'])
        self.dense2 = tf.keras.layers.Dense(CFG['dim'], use_bias=False)
        self.dropout3 = tf.keras.layers.Dropout(CFG['drop_rate'], noise_shape=(None,1,1))
        self.add3 = tf.keras.layers.Add()
        
        self.batch_norm4 = tf.keras.layers.BatchNormalization(momentum=CFG['bn_momentum']) 
        # self.layer_norm4 = tf.keras.layers.LayerNormalization(dtype=tf.bfloat16)  
        
    def call(self, enc_output, x2):
        mask = create_look_ahead_mask(tf.shape(x2)[1]) # Create mask
        
        # Base mha
        # x = self.batch_norm(inp2)
        # x2 = self.layer_norm(x2)
        x = self.mha1(q=x2, k=x2, v=x2, mask=mask)
        x = self.dropout1(x)
        x = self.add1([x, x2])
        
        # Cross mha
        base_out = x
        x = self.batch_norm1(x)
        y = self.batch_norm2(enc_output)
        # x = self.layer_norm1(x)
        # y = self.layer_norm2(enc_output)
        x = self.mha2(q=x, k=y, v=y, mask=None)
        x = self.dropout2(x)
        x = self.add2([x, x2])
        # x = self.add2([x, base_out])

        # Feed-forward
        cross_out = x
        x = self.batch_norm3(x)
        # x = self.layer_norm3(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout3(x)
        # x = self.add3([x, x2])
        x = self.add3([x, cross_out])

        x = self.batch_norm4(x)
        # x = self.layer_norm4(x)
        return x

class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__(name='encoder')
        self.mask_frames = tf.keras.layers.Masking(CFG['pad_frames'], input_shape=(None, CFG['max_len_frames'] ,CHANNELS))
        self.dense = tf.keras.layers.Dense(CFG['dim'], use_bias=False)
        self.batch_norm =  tf.keras.layers.BatchNormalization(momentum=CFG['bn_momentum'])
        self.conv1d_block1 = Conv1DBlock()
        self.conv1d_block2 = Conv1DBlock()
        self.conv1d_block3 = Conv1DBlock()
        self.transformer_base_block1 = TransformerBaseBlock()

        self.conv1d_block4 = Conv1DBlock()
        self.conv1d_block5 = Conv1DBlock()
        self.conv1d_block6 = Conv1DBlock()
        self.transformer_base_block2 = TransformerBaseBlock()
        
    def call(self, x):
        x = self.mask_frames(x)
        x = self.dense(x)
        x = self.batch_norm(x)
        
        # Conv1D + transformer block
        x = self.conv1d_block1(x)
        x = self.conv1d_block2(x)
        x = self.conv1d_block3(x)
        x = self.transformer_base_block1(x)
    
        x = self.conv1d_block4(x)
        x = self.conv1d_block5(x)
        x = self.conv1d_block6(x)
        x = self.transformer_base_block2(x)
        return x 

class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__(name='decoder')
        self.mask_phrase = tf.keras.layers.Masking(CFG['pad_phrase'], input_shape=(None, CFG['max_len_phrase']))
        self.positional_embedding = PositionalEmbedding()
        self.transformer_cross_block = TransformerCrossBlock()
    
    def call(self, encoder_out, x2):
        x = self.mask_phrase(x2)
        x = self.positional_embedding(x)
        x = self.transformer_cross_block(encoder_out,x)
        return x

def get_model(config, dtype):
    global CFG
    global DTYPE
    CFG = config
    DTYPE = dtype
    
    inp1 = tf.keras.layers.Input([CFG['max_len_frames'], CHANNELS], dtype=tf.float32, name='frames')
    inp2 = tf.keras.layers.Input([CFG['max_len_phrase']], dtype=tf.int32, name='phrase')

    # Encoder and decoder step
    encoder_out = Encoder()(inp1)
    decoder_out = Decoder()(encoder_out, inp2)

    # Final feed-forward
    out = tf.keras.layers.Dense(CFG['dim']*2,activation=None, name='pre_classifier')(decoder_out)
    out = tf.keras.layers.Dense(CFG['num_classes'], name='classifier')(out)
    
    model = tf.keras.models.Model(inputs=[inp1, inp2], outputs=out)
    return model