from tensorflow.keras import layers, Sequential, Model, mixed_precision
import numpy as np
from tensorflow.keras import losses, optimizers, callbacks, metrics
from tensorflow import GradientTape
from tensorflow import range as tfrange
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.math import argmax

class ModelConfigurator():
    def __init__(self, MAX_LEN, NUM_HEAD, FF_DIM, NUM_LAYERS, EMBED_DIM, LR):
        self.MAX_LEN = MAX_LEN
        self.NUM_HEAD = NUM_HEAD
        self.FF_DIM = FF_DIM
        self.NUM_LAYERS = NUM_LAYERS
        self.EMBED_DIM = EMBED_DIM
        self.LR = LR


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


loss_fn = losses.SparseCategoricalCrossentropy(
    reduction=losses.Reduction.NONE
)
loss_tracker = metrics.Mean(name="loss")

accuracy = metrics.SparseCategoricalAccuracy()


class MaskedLanguageModel(Model):
    def train_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        with GradientTape() as tape:
            predictions = self(features, training=True)
            loss = loss_fn(labels, predictions, sample_weight=sample_weight)
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        accuracy.update_state(labels, predictions, sample_weight=sample_weight)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        #gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute loss
        loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result(), "accuracy": accuracy.result()}
    
    def test_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None
        
        predictions = self(features, training=False)
        # Updates the metrics tracking the loss
        loss = loss_fn(labels, predictions, sample_weight=sample_weight)
        loss_tracker.update_state(loss, sample_weight=sample_weight)
        # Update the metrics.
        accuracy.update_state(labels, predictions, sample_weight=sample_weight)
        #perplexity.update_state(labels, predictions, sample_weight=sample_weight) # test
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"loss": loss_tracker.result(), "accuracy": accuracy.result()}
    
    @property
    def metrics(self):
        return [loss_tracker, accuracy]


def create_mlm(mc, vocab_size):
    inputs = layers.Input((mc.MAX_LEN,), dtype='int32')

    word_embeddings = layers.Embedding(
        vocab_size, mc.EMBED_DIM, name="word_embedding"
    )(inputs)
    position_embeddings = layers.Embedding(
        input_dim=mc.MAX_LEN,
        output_dim=mc.EMBED_DIM,
        weights=[get_pos_encoding_matrix(mc)],
        name="position_embedding",
    )(tfrange(start=0, limit=mc.MAX_LEN, delta=1))
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    for i in range(mc.NUM_LAYERS):
        encoder_output = att_module(mc, encoder_output, encoder_output, encoder_output, i)

    mlm_dense = layers.Dense(vocab_size, name="mlm_cls", activation="softmax")(
        encoder_output
    )

    mlm_model = MaskedLanguageModel(inputs, mlm_dense, name="masked_bert_model")

    optimizer = optimizers.Adam(learning_rate=mc.LR)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    mlm_model.compile(optimizer=optimizer, metrics=[accuracy])
    return mlm_model


def plot_history(hist):
    hist_df = pd.DataFrame(
        {'loss': hist.history['loss'],
        'accuracy': hist.history['accuracy'],
        'val_loss': hist.history['val_loss'],
        'val_accuracy': hist.history['val_accuracy']
        }
        )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    hist_df[['loss', 'val_loss']].plot(ax=ax1)
    hist_df[['accuracy', 'val_accuracy']].plot(ax=ax2)
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')
    plt.show()
