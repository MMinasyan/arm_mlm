from tensorflow.keras import layers, Sequential, Model
import numpy as np

class ModelConfigurator():
    def __init__(self, MAX_LEN, NUM_HEAD, FF_DIM, NUM_LAYERS, EMBED_DIM):
        self.MAX_LEN = MAX_LEN
        self.NUM_HEAD = NUM_HEAD
        self.FF_DIM = FF_DIM
        self.NUM_LAYERS = NUM_LAYERS
        self.EMBED_DIM = EMBED_DIM


def att_module(mc, query, key, value, i):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=mc.NUM_HEAD,
        key_dim=mc.EMBED_DIM // mc.NUM_HEAD,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value)
    attention_output = layers.Dropout(0.2, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = Sequential(
        [
            layers.Dense(mc.FF_DIM, activation="relu"),
            layers.Dense(mc.EMBED_DIM),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.15, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output


def get_pos_encoding_matrix(mc):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / mc.EMBED_DIM) for j in range(mc.EMBED_DIM)]
            if pos != 0
            else np.zeros(mc.EMBED_DIM)
            for pos in range(mc.MAX_LEN)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc