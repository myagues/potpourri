"""Text generation with Flax.
Using a basic recurrent neural network, train a language model for a few epochs and
generate text from an initial string.

Mainly based on the TF tutorial [Text generation with an RNN](https://www.tensorflow.org/tutorials/text/text_generation).
"""

import argparse
import functools
import os
import time

import jax

import numpy as np
import tensorflow as tf

from flax import linen as nn, optim
from flax.training import checkpoints, common_utils
from jax import numpy as jnp
from jax.config import config as jax_config

jax_config.enable_omnistaging()


def build_ds(path_to_file, batch_size, seq_length):
    """Build dataset.
    Args:
        batch_size: size of the output data batch
        seq_length: length of batch sequence
        path_to_file: path to the dataset text file
    Returns:
        dataset: dataset iterator with token IDs
        vocab: list of characters in the vocabulary
        idx2char: token IDs to character converter
        char2idx: character to token IDs converter
    """
    text = open(path_to_file, "rb").read().decode(encoding="utf-8")
    vocab = sorted(set(text))

    char2idx = {c: idx for idx, c in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_label(chunk):
        input_text = chunk[:-1]
        label_text = chunk[1:]
        return {"inputs": input_text, "labels": label_text}

    dataset = sequences.map(split_input_label).cache()
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, vocab, idx2char, char2idx


class RNN(nn.Module):
    @functools.partial(
        nn.transforms.scan, variable_broadcast="params", split_rngs={"params": False}
    )
    @nn.compact
    def __call__(self, carry, x):
        return nn.GRUCell()(carry, x)

    @staticmethod
    def initialize_carry(hidden_size):
        # use dummy key since default state init fn is just zeros.
        return nn.GRUCell.initialize_carry(jax.random.PRNGKey(0), (), hidden_size)


class charRNN(nn.Module):
    """charRNN module."""

    vocab_size: int
    hidden_size: int = 1024
    emb_dim: int = 256

    @nn.compact
    def __call__(self, inputs, carry=None):
        """Run the charRNN model.
        Args:
            inputs: input batch of character IDs
            carry: contains the previous cell state
        Returns:
            logits: output logits
            carry: output state to feed in the next step
        """
        x = inputs.astype("int32")
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.emb_dim)(x)

        rnn = RNN()
        if carry is None:
            carry = rnn.initialize_carry(self.hidden_size)

        carry, x = rnn(carry, x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits, carry


def get_initial_params(key, vocab_size, max_input_len=2048):
    """Creates a seq2seq model."""
    input_shape = jnp.ones((max_input_len,), jnp.int32)
    return charRNN(vocab_size).init({"params": key}, input_shape)["params"]


def cross_entropy_loss(logits, labels):
    """Returns cross-entropy loss."""
    onehot_labels = common_utils.onehot(labels, logits.shape[-1])
    xe = jnp.sum(onehot_labels * nn.log_softmax(logits), axis=-1)
    return -jnp.mean(xe)


@functools.partial(jax.jit, static_argnums=(3,))
def train_step(optimizer, batch, state, vocab_size):
    """Train one step."""
    inputs, labels = batch

    def loss_fn(params):
        v_char_rnn = jax.vmap(
            functools.partial(charRNN(vocab_size).apply, {"params": params})
        )
        logits, new_state = v_char_rnn(inputs, state)
        loss = cross_entropy_loss(logits, labels)
        return loss, (logits, new_state)

    aux, grad = jax.value_and_grad(loss_fn, has_aux=True)(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    loss, (logits, new_state) = aux
    return optimizer, new_state, loss


def train():
    args = parse_args()
    path_to_file = tf.keras.utils.get_file(
        "shakespeare.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
        cache_subdir=args.data_dir,
    )
    ds, vocab, *_ = build_ds(path_to_file, args.batch_size, args.seq_length)

    rng = jax.random.PRNGKey(args.seed)
    params = get_initial_params(rng, len(vocab), args.seq_length)
    optimizer = optim.Adam(learning_rate=args.lr).create(params)
    optimizer = checkpoints.restore_checkpoint(args.ckpt_dir, optimizer)

    for epoch in range(1, args.num_epochs + 1):
        state = None
        start = time.time()
        for batch in ds:
            batch = batch["inputs"]._numpy(), batch["labels"]._numpy()
            optimizer, state, loss = train_step(optimizer, batch, state, len(vocab))
        print(f"Epoch: {epoch:2d}, loss: {loss:.4f}, time: {time.time() - start:.4f}s")
        checkpoints.save_checkpoint(args.ckpt_dir, optimizer, epoch, keep=3)


@functools.partial(jax.jit, static_argnums=(4, 5))
def eval_fn(params, inputs, state, rng, vocab_size, temperature=1.0):
    """Predict characters in the sequence using the previous network states."""
    preds, new_state = charRNN(vocab_size).apply({"params": params}, inputs, state)
    # using a categorical distribution to predict the character returned by the model
    preds = preds / temperature
    predicted_id = jax.random.categorical(rng, preds)[-1]
    return predicted_id, new_state


def generate_text():
    """Generate sequence of text with a saved model."""
    args = parse_args()
    path_to_file = os.path.join(args.data_dir, "shakespeare.txt")
    text = open(path_to_file, "rb").read().decode(encoding="utf-8")
    vocab = sorted(set(text))

    char2idx = {c: idx for idx, c in enumerate(vocab)}
    idx2char = np.array(vocab)

    # Converting our start string to numbers (vectorizing)
    inputs = np.array([char2idx[s] for s in args.text_input])
    text_generated = []

    rng = jax.random.PRNGKey(args.seed)
    params = get_initial_params(rng, len(vocab), args.seq_length)
    optimizer = optim.Adam(learning_rate=args.lr).create(params)
    optimizer = checkpoints.restore_checkpoint(args.ckpt_dir, optimizer)
    state = None

    for _ in range(args.num_generate):
        rng, rng1 = jax.random.split(rng)
        predicted_id, state = eval_fn(
            optimizer.target, inputs, state, rng1, len(vocab), 1.0
        )
        text_generated.append(idx2char[predicted_id])
        inputs = predicted_id[None]
    print(args.text_input + "".join(text_generated))


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Train and eval a text generation model."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Directory where data file will be saved.",
    )
    parser.add_argument(
        "--ckpt_dir",
        required=True,
        type=str,
        help="Directory where the ckpt file will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Size of the batch to use during training.",
    )
    parser.add_argument(
        "--emb_dim",
        default=256,
        type=int,
        help="Embedding dimension.",
    )
    parser.add_argument(
        "--rnn_units",
        default=1024,
        type=int,
        help="RNN cell hidden units.",
    )
    parser.add_argument(
        "--seq_length",
        default=100,
        type=int,
        help="Length of the batch sequence.",
    )
    parser.add_argument(
        "--num_epochs",
        default=30,
        type=int,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed.",
    )
    parser.add_argument(
        "--text_input",
        default="ROMEO: ",
        type=str,
        help="Input text for generation.",
    )
    parser.add_argument(
        "--num_generate",
        default=100,
        type=int,
        help="Number of characters to generate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train()
    generate_text()
