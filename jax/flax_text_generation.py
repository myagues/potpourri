"""Text generation with Flax.
Using a basic recurrent neural network, train a language model for a few epochs and
generate text from an initial strig.

Works with stateful scopes to share network state between batches, instead of explicitly
feeding the state of the previous batch.

Mainly based on the TF tutorial [Text generation with an RNN](https://www.tensorflow.org/tutorials/text/text_generation).
"""

import argparse
import os
import tempfile
import time

import jax

import numpy as np
import tensorflow as tf

from flax import jax_utils, nn, optim
from flax.training import checkpoints, common_utils
from functools import partial
from jax import numpy as jnp
from typing import Dict, List, Tuple

State = nn.Collection


def build_ds(
    batch_size: int, seq_length: int, path_to_file: str,
) -> Tuple[tf.data.Dataset, List[str], jnp.ndarray, Dict[str, int]]:
    """Build dataset.
    Args:
        batch_size: size of the output data batch.
        seq_length: length of batch sequence.
        path_to_file: path to the dataset text file.
    Returns:
        dataset: dataset iterator with token IDs.
        vocab: list of characters in the vocabulary.
        idx2char: token IDs to character converter.
        char2idx: character to token IDs converter.
    """
    text = open(path_to_file, "rb").read().decode(encoding="utf-8")
    vocab = sorted(set(text))

    char2idx = {c: idx for idx, c in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk: tf.Tensor) -> Dict[str, tf.Tensor]:
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return {"inputs": input_text, "targets": target_text}

    dataset = sequences.map(split_input_target).cache()
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, vocab, idx2char, char2idx


class Embed(nn.Module):
    """Embedding Module.
    A parameterized function from integers [0, n) to d-dimensional vectors.
    """

    def apply(
        self,
        inputs: jnp.ndarray,
        num_embeddings: int,
        features: int,
        mode: str = "input",
        emb_init: nn.initializers = nn.initializers.normal(stddev=1.0),
    ) -> jnp.ndarray:
        """Applies Embed module.
        Args:
            inputs: input data
            num_embeddings: number of embedding
            features: size of the embedding dimension
            mode: either 'input' or 'output' -> to share input/output embedding
            emb_init: embedding initializer
        Returns:
            output which is embedded input data
        """
        embedding = self.param("embedding", (num_embeddings, features), emb_init)
        if mode == "input":
            if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
                raise ValueError("Input type must be an integer or unsigned integer.")
            return jnp.take(embedding, inputs, axis=0)
        if mode == "output":
            return jnp.einsum("bld,vd->blv", inputs, embedding)


class RNN(nn.Module):
    """RNN module."""

    def apply(
        self, inputs: jnp.ndarray, cell_type: nn.Module, hidden_size: int,
    ) -> jnp.ndarray:
        """Applies RNN module."""
        batch_size = inputs.shape[0]
        cell = cell_type.shared(name="cell")
        partial_func = partial(jax_utils.scan_in_dim, body_fn=cell, xs=inputs, axis=1)
        init = cell.initialize_carry(nn.make_rng(), (batch_size,), hidden_size)

        if self.is_stateful():
            state = self.state("state")
            if self.is_initializing():
                state.value = init
            last_state, y = partial_func(init=state.value)
            if not self.is_initializing():
                state.value = last_state
        else:
            _, y = partial_func(init=init)
        return y


class charRNN(nn.Module):
    """charRNN module."""

    def apply(
        self,
        inputs: jnp.ndarray,
        vocab_size: int,
        emb_dim: int = 256,
        cell_type: nn.Module = nn.GRUCell,
        hidden_size: int = 1024,
    ) -> jnp.ndarray:
        """Run the charRNN model.
        Args:
            inputs: input batch of character IDs.
            vocab_size: size of the vocabulary.
            emb_dim: embedding size.
            cell_type: type of RNN cell.
            hidden_size: RNN cell size.
        Returns:
            logits: output logits.
        """
        x = inputs.astype("int32")
        x = Embed(x, num_embeddings=vocab_size, features=emb_dim, name="embed")
        x = RNN(x, cell_type=cell_type, hidden_size=hidden_size)
        logits = nn.Dense(
            x,
            vocab_size,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )
        return logits


def create_model(
    vocab_size: int, batch_size: int, seq_length: int, emb_dim: int, hidden_size: int
) -> Tuple[nn.Model, nn.Collection]:
    """Creates and initializes charRNN model."""
    with nn.stateful() as init_state:
        partial_model = charRNN.partial(
            vocab_size=vocab_size, emb_dim=emb_dim, hidden_size=hidden_size
        )
        _, initial_params = partial_model.init_by_shape(
            nn.make_rng(), [((batch_size, seq_length), jnp.float32)],
        )
        model = nn.Model(partial_model, initial_params)
    return model, init_state


def create_optimizer(model: nn.Model, learning_rate: float) -> optim.OptimizerDef:
    """Creates an Adam optimizer for @model."""
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    optimizer = optimizer_def.create(model)
    return optimizer


def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Returns cross-entropy loss."""
    onehot_targets = common_utils.onehot(targets, logits.shape[-1])
    xe = jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
    return -jnp.mean(xe)


@jax.jit
def train_step(
    optimizer: optim.OptimizerDef,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jnp.ndarray,
    state: State,
) -> Tuple[optim.OptimizerDef, State, jnp.ndarray, jnp.ndarray]:
    """Train one step."""
    inputs, targets = batch

    def loss_fn(model: nn.Model) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Compute cross-entropy loss."""
        with nn.stochastic(rng), nn.stateful(state) as new_state:
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            return loss, (new_state, logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_state, logits)), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, new_state, loss, logits


def train() -> None:
    args = parse_args()
    path_to_file = tf.keras.utils.get_file(
        "shakespeare.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
        cache_subdir=args.data_dir,
    )
    ds, vocab, *_ = build_ds(args.batch_size, args.seq_length, path_to_file)

    with nn.stochastic(jax.random.PRNGKey(args.seed)):
        model, state = create_model(
            len(vocab), args.batch_size, args.seq_length, args.emb_dim, args.rnn_units
        )
        optimizer = create_optimizer(model, args.lr)
        del model
        for epoch in range(1, args.num_epochs + 1):
            start = time.perf_counter()
            for batch in ds:
                inputs, targets = batch["inputs"]._numpy(), batch["targets"]._numpy()
                optimizer, state, loss, _ = train_step(
                    optimizer, (inputs, targets), nn.make_rng(), state
                )
            checkpoints.save_checkpoint(args.ckpt_dir, optimizer.target, epoch)
            print(
                f"Epoch: {epoch:2d}, loss: {loss:.4f}, "
                f"time: {time.perf_counter() - start:.4f}"
            )


@jax.jit
def eval_fn(
    model: nn.Model,
    input_eval: jnp.ndarray,
    state: State,
    rng: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, State]:
    """Predict characters in the sequence using the previous network states."""
    with nn.stateful(state) as new_state:
        predictions = model(input_eval)
        # using a categorical distribution to predict the character returned by the model
        predictions = predictions[0] / temperature
        predicted_id = jax.random.categorical(nn.make_rng(), predictions)[-1]
    return predicted_id, new_state


def generate_text() -> None:
    """Generate sequence of text with a saved model."""
    args = parse_args()
    data_path = os.path.join(args.data_dir, "shakespeare.txt")
    text = open(data_path, "rb").read().decode(encoding="utf-8")
    vocab = sorted(set(text))

    char2idx = {c: idx for idx, c in enumerate(vocab)}
    idx2char = np.array(vocab)

    # Converting our start string to numbers (vectorizing)
    input_eval = jnp.array([char2idx[s] for s in args.text_input])
    input_eval = input_eval[jnp.newaxis, ...]

    text_generated = []

    with nn.stochastic(jax.random.PRNGKey(args.seed)):
        model, state = create_model(
            len(vocab), args.batch_size, args.seq_length, args.emb_dim, args.rnn_units
        )
        model = checkpoints.restore_checkpoint(args.ckpt_dir, model)
        for _ in range(args.num_generate):
            predicted_id, state = eval_fn(model, input_eval, state, 0)
            input_eval = jnp.reshape(predicted_id, (-1, 1))
            text_generated.append(idx2char[predicted_id])
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
        "--emb_dim", default=256, type=int, help="Embedding dimension.",
    )
    parser.add_argument(
        "--rnn_units", default=1024, type=int, help="RNN cell hidden units.",
    )
    parser.add_argument(
        "--seq_length", default=100, type=int, help="Length of the batch sequence.",
    )
    parser.add_argument(
        "--num_epochs", default=30, type=int, help="Number of epochs.",
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--seed", default=1234, type=int, help="Random seed.",
    )
    parser.add_argument(
        "--text_input", default="ROMEO: ", type=str, help="Input text for generation.",
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
