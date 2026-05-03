# Interactive MNIST

I saw a website like this in a video on Instagram, but could not find the source, so I decided to implement it myself as a weekend project.

Live site: https://eduardomineo.github.io/interactive_mnist/

Train a small Keras neural network on MNIST and use it in a static browser UI where drawings are classified live. The model uses two 25-neuron tanh hidden layers. The visualization shows the 28x28 input grid, hidden activations, output probabilities for digits 0-9, and illustrative layer connections.

## Requirements

- Python 3.8, 3.9, or 3.10
- A recent browser
- Optional: Node.js, if you prefer serving the static app with `npx`

## Train And Export

Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependency pins target TensorFlow 2.11.0. TensorFlow.js 4.4.0 needs an older JAX API during export, so keep the pinned JAX packages unless the exporter is upgraded.

Train the model and export the browser model:

```bash
python train_keras.py
```

This writes the Keras model to `model/mnist_model.keras` and the TensorFlow.js model to `app/model/`.
During training, the script keeps the weights with the best validation loss and uses those weights for test evaluation and export.

Useful options:

```bash
python train_keras.py --epochs 50 --batch-size 128
python train_keras.py --skip-tfjs-export
```

## Run Locally

After exporting the model, serve the `app/` directory:

```bash
python -m http.server 8000 --directory app
```

Open `http://localhost:8000` in your browser.

You can also use:

```bash
npx serve app
```

## Deploy

The browser app is static after export. Deploy the contents of `app/` to any static host, including GitHub Pages, Netlify, Vercel, S3, or a simple web server.

The deployed `app/` directory must include:

- `index.html`
- `styles.css`
- `main.js`
- `model/model.json`
- the generated model weight shard files in `model/`

## Project Layout

- `train_keras.py`: trains the Keras model and exports it for the browser
- `app/`: static browser UI
- `requirements.txt`: Python dependencies for training and export
