import tensorflow as tf
from ...utils.preprocessing.preprocess_features import *

class ECA(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_average = tf.keras.layers.GlobalAveragePooling1D()
        self.conv = tf.keras.layers.Conv1D(1, CFG['ksize'], strides=1, padding="same", use_bias=False) #5?
    
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
        self.dense1 = tf.keras.layers.Dense(CFG['encoder_dim']*CFG['expand'], use_bias=True, activation=CFG['activation'])
        self.causal_dw_conv1d = CausalDWConv1D()
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=CFG['bn_momentum'])
        self.eca = ECA()
        self.dense2 = tf.keras.layers.Dense(CFG['encoder_dim'], use_bias=True)
        self.dropout = tf.keras.layers.Dropout(CFG['drop_rate'], noise_shape=(None,1,1))
        
    def call(self, inp1):
        x = self.dense1(inp1)
        x = self.causal_dw_conv1d(x)
        x = self.batch_norm(x)
        x = self.eca(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = x + inp1 # skip connection
        return x

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dim = CFG['mha_dim']
        self.scale = self.dim ** -0.5
        self.num_heads = CFG['num_heads']
        self.qkv = tf.keras.layers.Dense(3 * CFG['mha_dim'], use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(CFG['mha_dropout'])
        self.proj = tf.keras.layers.Dense(CFG['mha_dim'], use_bias=False)

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x
    
# class MultiHeadAttention(tf.keras.layers.Layer):
#     def __init__(self, dim, **kwargs):   # assert dim % self.num_heads == 0
#         super().__init__()
#         self.num_heads = CFG['num_heads']
#         self.dim = dim
#         self.depth = dim // self.num_heads

#         self.wq = tf.keras.layers.Dense(dim)
#         self.wk = tf.keras.layers.Dense(dim)
#         self.wv = tf.keras.layers.Dense(dim)

#         self.wo = tf.keras.layers.Dense(dim)
#         self.mha_dropout = tf.keras.layers.Dropout(CFG['drop_rate'])
        
#     def split_heads(self, x, batch_size):
#         """Split the last dimension into (num_heads, depth). Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
#         """
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
#         return tf.transpose(x, perm=[0, 2, 1, 3])

#     def call(self, q, k, v, mask=None):
#         batch_size = tf.shape(q)[0]

#         q = self.wq(q)  # (batch_size, seq_len, d_model)
#         k = self.wk(k)  # (batch_size, seq_len, d_model)
#         v = self.wv(v)  # (batch_size, seq_len, d_model)

#         q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
#         k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
#         v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

#         # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
#         # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
#         scaled_attention = scaled_dot_product_attention(q, k, v, mask)
#         scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
#         concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dim))  # (batch_size, seq_len_q, d_model)

#         output = self.wo(concat_attention)  # (batch_size, seq_len_q, d_model)
#         output = self.mha_dropout(output)
#         return output
    
class TransformerBaseBlock(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
    def build(self, input_shape):
        self.bn_1s = []
        self.mhas = []
        self.bn_2s = []
        self.mlps = []
        for i in range(CFG['len_transformer_base_block']):
            self.bn_1s.append(tf.keras.layers.BatchNormalization(momentum=CFG['bn_momentum']))
            self.mhas.append(MultiHeadSelfAttention())
            self.bn_2s.append(tf.keras.layers.BatchNormalization(momentum=CFG['bn_momentum']))
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(CFG['encoder_dim']*CFG['expand'], activation=CFG['activation'], use_bias=False),
                tf.keras.layers.Dropout(CFG['drop_rate']),
                tf.keras.layers.Dense(CFG['encoder_dim'], use_bias=False),
            ]))
        
    def call(self, x1):
        for bn_1, mha, bn_2, mlp in zip(self.bn_1s, self.mhas, self.bn_2s, self.mlps):
            x = bn_1(x1 + mha(x1))
            x = bn_2(x + mlp(x))
        return x

    
class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__(name='encoder')
        
    def build(self, input_shape):
        self.mask_frames = tf.keras.layers.Masking(CFG['pad_frames'], input_shape=(None, CFG['max_len_frames'] ,CHANNELS))
        self.mlp = tf.keras.layers.Dense(CFG['encoder_dim'], use_bias=False)
        self.bn_inp =  tf.keras.layers.BatchNormalization(momentum=CFG['bn_momentum'])
        
        self.conv_blocks = [] # [Conv1DBlock() for _ in range(CFG['len_conv1d_blocks_encoder'])]
        self.transformer_base_blocks = []
        for i in range(CFG['n_encoder_blocks']):
            self.conv_blocks.append([Conv1DBlock() for _ in range(CFG['len_conv1d_blocks_encoder'])])
            self.transformer_base_blocks.append(TransformerBaseBlock())
            
    def call(self, x):
        x = self.mask_frames(x)
        x = self.mlp(x)
        x = self.bn_inp(x)
        
        for conv_block, transformer_base_block in zip(self.conv_blocks, self.transformer_base_blocks):
            for conv in conv_block:
                x = conv(x)
            x = transformer_base_block(x)
        return x 

class LateDropout(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name='LateDropout')
        self.dropout = tf.keras.layers.Dropout(CFG['late_drop_rate'], noise_shape=None)
      
    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        x = tf.cond(self._train_counter < CFG['late_drop_start_epoch'], lambda:inputs, lambda:self.dropout(inputs, training=training))
        if training:
            self._train_counter.assign_add(1)
        return x

# class GeMPoolingLayer(tf.keras.layers.Layer):
#     def __init__(self, p=1., train_p=False, mixed_prec=True):
#         super().__init__()
#         if train_p:
#             if mixed_prec:
#                 self.p = tf.Variable(p, dtype=tf.float16)
#             else:
#                 self.p = tf.Variable(p, dtype=tf.float32)
#         else:
#             self.p = p
#         self.eps = 1e-7

#     def call(self, inputs: tf.Tensor, **kwargs):
#         inputs = tf.clip_by_value(inputs, clip_value_min=self.eps, clip_value_max=tf.reduce_max(inputs))
#         inputs = tf.pow(inputs, self.p)
#         inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
#         inputs = tf.pow(inputs, 1./self.p)
#         return inputs
    
def get_model(config, dtype):
    global CFG
    global DTYPE
    CFG = config
    DTYPE = dtype
    
    inp1 = tf.keras.layers.Input([CFG['max_len_frames'], CHANNELS], dtype=tf.float32, name='frames')

    # Encoder and decoder step
    encoder_out = Encoder()(inp1)

    # Final feed-forward
    out = tf.keras.layers.Dense(CFG['encoder_dim']*2,activation=None, name='pre_classifier')(encoder_out)
    # out = tf.keras.layers.AveragePooling1D(CFG['avg_pool_size'], name = 'avg_pooling1d')(out)
    out = LateDropout()(out)
    out = tf.keras.layers.Dense(CFG['num_classes'], name='classifier')(out)
    
    model = tf.keras.models.Model(inputs=inp1, outputs=out)
    return model