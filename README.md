<div align="center">

# 🔤 Character Recognition Neural Network

**A 35-class character classifier built from absolute scratch — no TensorFlow, no PyTorch, no Keras. Just NumPy.**

![Python](https://img.shields.io/badge/python-3.13-blue)
![NumPy](https://img.shields.io/badge/built%20with-NumPy%20only-013243?logo=numpy&logoColor=white)
![Test Accuracy](https://img.shields.io/badge/test%20accuracy-84.9%25-success)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

</div>

---

Every piece of this network — forward propagation, backpropagation, activations,
loss, gradient descent — is hand-derived and hand-verified against small
numeric examples before ever touching real data. No black boxes.

## ✨ Highlights

- 🧠 **Fully custom neural net** — forward/backward propagation, ReLU, Softmax,
  cross-entropy loss, and mini-batch gradient descent, all implemented from
  first principles
- 🎨 **Self-generated dataset** — 14,000 synthetic character images rendered
  from 30 filtered system fonts, with rotation/translation/noise augmentation
- 🔍 **Debugged like a real project** — includes a documented dead-ReLU
  collapse, diagnosed and fixed, not just a clean first-try result
- 📊 **84.9% test accuracy** on 35 classes (A-Z + 1-9), with a full confusion
  matrix and human-readable error analysis
- ✅ **Every function verified by hand** — math checked against tiny toy
  networks before scaling up to the real 784→128→35 architecture

---

## 📐 Architecture

```
   Input                Hidden                 Output
 (784, image)  ──▶  (128, ReLU)  ──▶  (35, Softmax)
```

| Layer | Size | Activation | Init |
|:--|:--:|:--:|:--|
| Input | 784 | — | — |
| Hidden | 128 | ReLU | He (`√(2/n_in)`) |
| Output | 35 | Softmax | Xavier (`√(1/n_in)`) |

---

## 🗂️ Dataset

| | |
|---|---|
| **Classes** | 35 (A–Z, 1–9) |
| **Fonts used** | 30 (filtered from 191 system font files) |
| **Total images** | 14,000 (400/class) |
| **Image size** | 28×28 grayscale, normalized [0, 1] |
| **Augmentation** | ±12° rotation, ±4px shift, Gaussian noise (σ=8) |
| **Split** | 9,800 train / 2,100 val / 2,100 test (stratified) |

No external dataset (MNIST, EMNIST) was used — neither matches this project's
exact 35-class set, so the dataset is fully self-generated and reproducible.

---

## 🏋️ Training

```python
train(X_train, y_train, X_val, y_val,
      layer_dims=[784, 128, 35],
      epochs=80, batch_size=128, learning_rate=0.05, seed=1)
```

<details>
<summary><b>🐛 The debugging story (click to expand)</b></summary>

<br>

**Attempt 1 — `learning_rate=0.5`:** training flatlined instantly. Validation
accuracy sat at exactly 1/35 (random-guess baseline) for all 30 epochs.
A dead-neuron check revealed **100% of hidden neurons permanently inactive**
— the large learning rate pushed every neuron's pre-activation negative on
the very first update, zeroing its ReLU gradient forever.

**Attempt 2 — `learning_rate=0.05`:** collapse fixed, but validation accuracy
bounced wildly between ~30% and ~75% epoch to epoch, even as training loss
fell smoothly — a sign of noisy gradients from small batches.

**Attempt 3 — `batch_size=64 → 128`:** steadier gradient estimates resolved
the instability. Train and validation loss now track closely, converging
smoothly across 80 epochs.

</details>

---

## 📊 Results

| Metric | Score |
|---|:--:|
| **Best validation accuracy** | 84.95% (epoch 78/80) |
| **Final test accuracy** | **84.9%** |

The close match between validation and test accuracy indicates the network
generalizes well rather than overfitting.

A full 35×35 confusion matrix (in `notebook/eval.ipynb`) shows a strong
diagonal with a handful of specific, explainable confusions — not random noise.

---

## 🔬 Error analysis

Five misclassifications, pulled from the confusion matrix's hottest
off-diagonal cells:

| True | Predicted | Why |
|:--:|:--:|---|
| `J` | `I` | J's small bottom hook — its only distinguishing feature — is faint and easily lost to augmentation noise |
| `1` | `I` | Near-identical vertical-stroke glyphs in this particular font |
| `S` | `5` | Shared curve structure — bends the same way top and bottom |
| `8` | `B` | Both are two stacked closed loops; the subtler flat-spine vs. symmetric-loop difference is under-weighted |
| `R` | `B` | A decorative font's flourished leg creates an accidental second loop |

Every one of these is a visually reasonable mix-up to a human eye too — the
errors trace back to genuine font-level ambiguity, not a broken model.

---

## 📁 Project structure

```
character-recognition-nn/
├── src/
│   ├── NN_model.py        # core network: forward/backward prop, loss, etc.
│   └── NN_train.py        # training loop, batching, accuracy
├── notebook/
│   ├── synthds_generation1.ipynb   # dataset generation
│   ├── model_architecture.ipynb    # architecture + hand-verified math
│   ├── NN_train.ipynb              # training runs
│   └── eval.ipynb                  # evaluation, confusion matrix, errors
├── data/
│   ├── character_dataset.npz       # generated train/val/test split
│   └── trained_params.npz          # final trained weights
├── report/                          # written report
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

```bash
git clone <repo-url>
cd character-recognition-nn

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## ▶️ Usage

Run the notebooks in `notebook/`, in order:

1. **`synthds_generation1.ipynb`** — generates and saves the dataset
2. **`model_architecture.ipynb`** — defines and hand-verifies every function
3. **`NN_train.ipynb`** — trains the model, saves weights to `data/`
4. **`eval.ipynb`** — loads the trained model, evaluates, analyzes errors

**Or skip straight to inference** with the already-trained model:

```python
import numpy as np
from src.NN_model import forward_propagation

loaded = np.load('data/trained_params.npz')
params = {k: loaded[k] for k in loaded.files}

A2, _ = forward_propagation(X.T, params)   # X: (n_samples, 784)
predictions = A2.argmax(axis=0)
```

---

<div align="center">

Built as part of an AI/ML internship assignment — every layer, every gradient,
implemented and verified by hand.

</div>
