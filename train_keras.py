from __future__ import annotations

import argparse
import pathlib
import shutil

import numpy as np
import tensorflow as tf


MODEL_DIR = pathlib.Path("model")
KERAS_MODEL_PATH = MODEL_DIR / "mnist_model.keras"
BEST_WEIGHTS_PATH = MODEL_DIR / "mnist_best.weights.h5"
TFJS_MODEL_DIR = pathlib.Path("app") / "model"


def build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(784,), name="pixels"),
            tf.keras.layers.Dense(25, activation="tanh", name="hidden_1"),
            tf.keras.layers.Dense(25, activation="tanh", name="hidden_2"),
            tf.keras.layers.Dense(10, activation="softmax", name="output"),
        ],
        name="mnist_tanh_visualizer",
    )


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape((-1, 784)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 784)).astype("float32") / 255.0

    return x_train, y_train, x_test, y_test


def export_tfjs_model(model: tf.keras.Model) -> None:
    import tensorflowjs as tfjs

    if TFJS_MODEL_DIR.exists():
        shutil.rmtree(TFJS_MODEL_DIR)

    tfjs.converters.save_keras_model(model, str(TFJS_MODEL_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the MNIST visualization model."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--skip-tfjs-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(7)

    x_train, y_train, x_test, y_test = load_data()
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_WEIGHTS_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        callbacks=[checkpoint],
    )
    model.load_weights(BEST_WEIGHTS_PATH)
    print(f"loaded best validation-loss weights from {BEST_WEIGHTS_PATH}")
    BEST_WEIGHTS_PATH.unlink()

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"test_loss={loss:.4f} test_accuracy={accuracy:.4f}")

    model.save(KERAS_MODEL_PATH)
    print(f"saved Keras model to {KERAS_MODEL_PATH}")

    if not args.skip_tfjs_export:
        export_tfjs_model(model)
        print(f"exported TensorFlow.js model to {TFJS_MODEL_DIR}")


if __name__ == "__main__":
    main()
