"""Train a small MLP (1 hidden layer, ReLU) on frozen SAM embeddings to
classify a grid cell as particle vs. background -- replaces the logistic-
regression version (linear decision boundary; see README.md's "训练轻量
分类头" v2 section for why a nonlinear boundary was worth trying: held-out
validation precision improved with more data but recall stayed the
bottleneck, and a linear model may just not separate the classes well).

Plain numpy forward/backward pass + Adam, no new ML dependency (matches
this project's pinned/minimal-dependency style, see requirements.txt).

Usage:
    python scripts/train_particle_classifier.py
"""

import os

import numpy as np

DATA_PATH = "data/datasets/particle_classifier_train.npz"
OUT_PATH = "models/particle_classifier.npz"

HIDDEN_UNITS = 64
LR = 1e-3
L2 = 1e-4
EPOCHS = 60
BATCH_SIZE = 2048
VAL_FRACTION = 0.2
SEED = 0
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def forward(X, params):
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.maximum(z1, 0)  # ReLU
    z2 = a1 @ params["W2"] + params["b2"]
    p = sigmoid(z2).ravel()
    return p, a1, z1


def main():
    data = np.load(DATA_PATH, allow_pickle=True)
    features, labels, image_ids = data["features"], data["labels"], data["image_ids"]
    print(f"loaded {len(labels)} samples ({labels.sum()} pos, {(labels==0).sum()} neg), feature dim {features.shape[1]}")

    # split by image, not by sample, so no particle's cells leak across
    # train/val -- otherwise val accuracy would be optimistic (same image's
    # background texture on both sides)
    unique_ids = sorted(set(image_ids.tolist()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_ids)
    n_val_images = max(1, int(len(unique_ids) * VAL_FRACTION))
    val_ids = set(unique_ids[:n_val_images])
    train_mask = np.array([iid not in val_ids for iid in image_ids])

    X_train = features[train_mask].astype(np.float64)
    y_train = labels[train_mask].astype(np.float64)
    X_val = features[~train_mask].astype(np.float64)
    y_val = labels[~train_mask].astype(np.float64)
    print(f"train: {len(y_train)} samples from {len(unique_ids) - n_val_images} images; "
          f"val: {len(y_val)} samples from {n_val_images} images")

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std

    n_features = X_train_n.shape[1]
    init_rng = np.random.default_rng(SEED)
    params = {
        "W1": init_rng.normal(0, np.sqrt(2.0 / n_features), size=(n_features, HIDDEN_UNITS)),
        "b1": np.zeros(HIDDEN_UNITS),
        "W2": init_rng.normal(0, np.sqrt(2.0 / HIDDEN_UNITS), size=(HIDDEN_UNITS, 1)),
        "b2": np.zeros(1),
    }
    adam_m = {k: np.zeros_like(v) for k, v in params.items()}
    adam_v = {k: np.zeros_like(v) for k, v in params.items()}

    # inverse-frequency class weights so the loss doesn't just learn "always
    # predict background" if negatives dominate
    n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
    w_pos = len(y_train) / (2 * n_pos)
    w_neg = len(y_train) / (2 * n_neg)

    n = len(y_train)
    t = 0
    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            Xb, yb = X_train_n[idx], y_train[idx]
            sw = np.where(yb == 1, w_pos, w_neg)
            bs = len(yb)

            p, a1, z1 = forward(Xb, params)

            dz2 = ((p - yb) * sw / bs).reshape(-1, 1)
            grads = {
                "W2": a1.T @ dz2 + L2 * params["W2"],
                "b2": dz2.sum(axis=0),
            }
            da1 = dz2 @ params["W2"].T
            dz1 = da1 * (z1 > 0)
            grads["W1"] = Xb.T @ dz1 + L2 * params["W1"]
            grads["b1"] = dz1.sum(axis=0)

            t += 1
            for name in params:
                adam_m[name] = ADAM_BETA1 * adam_m[name] + (1 - ADAM_BETA1) * grads[name]
                adam_v[name] = ADAM_BETA2 * adam_v[name] + (1 - ADAM_BETA2) * (grads[name] ** 2)
                m_hat = adam_m[name] / (1 - ADAM_BETA1**t)
                v_hat = adam_v[name] / (1 - ADAM_BETA2**t)
                params[name] -= LR * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

            eps_log = 1e-8
            epoch_loss += -np.sum(sw * (yb * np.log(p + eps_log) + (1 - yb) * np.log(1 - p + eps_log)))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}: avg weighted loss={epoch_loss / n:.4f}")

    def evaluate(X, y, name):
        p, _, _ = forward(X, params)
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        acc = (tp + tn) / len(y)
        print(f"{name}: acc={acc:.3f} precision={precision:.3f} recall={recall:.3f} "
              f"(tp={tp} fp={fp} fn={fn} tn={tn})")

    evaluate(X_train_n, y_train, "train")
    evaluate(X_val_n, y_val, "val  ")

    os.makedirs("models", exist_ok=True)
    np.savez(OUT_PATH, W1=params["W1"], b1=params["b1"], W2=params["W2"], b2=params["b2"], mean=mean, std=std)
    print(f"saved MLP classifier to {OUT_PATH}")


if __name__ == "__main__":
    main()
