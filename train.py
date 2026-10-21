import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, f1_score

import data

SEED = 42
EPOCHS = 40
BATCH = 32
LR = 3e-4

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs("results", exist_ok=True)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(128, 1)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.relu(self.bn4(self.conv4(x)))
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        x = self.drop(x)
        return self.fc(x).squeeze(1)


def augment(x):
    # small random shifts, no flips because left and right matter in a chest film
    dx, dy = np.random.randint(-6, 7, size=2)
    x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(2, 3))
    # random brightness and contrast jitter
    scale = 1 + 0.2 * (torch.rand(x.shape[0], 1, 1, 1) - 0.5)
    shift = 0.2 * (torch.rand(x.shape[0], 1, 1, 1) - 0.5)
    return x * scale + shift


def to_tensor(X):
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) / 255.0
    return (X - 0.5) / 0.25


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(X[i:i + 128]) for i in range(0, len(X), 128)])
    p = torch.sigmoid(logits).numpy()
    return p


def main():
    X, y, patient = data.load()
    print("images", X.shape, "covid positives", int(y.sum()))

    # split by patient so the same person never appears in train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr, te = next(gss.split(X, y, groups=patient))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr2, va = next(gss2.split(X[tr], y[tr], groups=patient[tr]))
    va = tr[va]
    tr = tr[tr2]
    print("train", len(tr), "val", len(va), "test", len(te))

    # baseline: logistic regression on raw pixels
    flat = X.reshape(len(X), -1) / 255.0
    lr = LogisticRegression(max_iter=2000, C=0.01)
    lr.fit(flat[tr], y[tr])
    p_lr = lr.predict_proba(flat[te])[:, 1]
    auc_lr = roc_auc_score(y[te], p_lr)
    print("baseline logistic regression test auc %.3f" % auc_lr)

    Xtr, Xva, Xte = to_tensor(X[tr]), to_tensor(X[va]), to_tensor(X[te])
    ytr = torch.tensor(y[tr], dtype=torch.float32)

    model = SmallCNN()
    pos_weight = torch.tensor(float((len(ytr) - ytr.sum()) / ytr.sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    history = {"train_loss": [], "val_auc": []}
    best_auc, best_state = 0, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb = augment(Xtr[idx])
            opt.zero_grad()
            loss = loss_fn(model(xb), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        p_va = evaluate(model, Xva, y[va])
        auc_va = roc_auc_score(y[va], p_va)
        history["train_loss"].append(total / len(Xtr))
        history["val_auc"].append(auc_va)
        print("epoch %2d  loss %.3f  val auc %.3f" % (ep + 1, total / len(Xtr), auc_va))
        if auc_va > best_auc:
            best_auc = auc_va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, "results/cnn.pt")

    p_te = evaluate(model, Xte, y[te])
    auc = roc_auc_score(y[te], p_te)
    pred = (p_te > 0.5).astype(int)
    cm = confusion_matrix(y[te], pred)
    f1 = f1_score(y[te], pred)
    print("cnn test auc %.3f  f1 %.3f" % (auc, f1))
    print("confusion matrix (rows true, cols pred)\n", cm)

    metrics = {
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "baseline_logreg_auc": float(auc_lr),
        "cnn_test_auc": float(auc), "cnn_test_f1": float(f1),
        "confusion_matrix": cm.tolist(), "best_val_auc": float(best_auc),
    }
    json.dump(metrics, open("results/metrics.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(history["train_loss"])
    ax[0].set_title("train loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(history["val_auc"])
    ax[1].set_title("validation AUC")
    ax[1].set_xlabel("epoch")
    fpr, tpr, _ = roc_curve(y[te], p_te)
    fpr2, tpr2, _ = roc_curve(y[te], p_lr)
    ax[2].plot(fpr, tpr, label="CNN auc %.2f" % auc)
    ax[2].plot(fpr2, tpr2, label="logreg auc %.2f" % auc_lr)
    ax[2].plot([0, 1], [0, 1], "k:")
    ax[2].set_title("test ROC")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig("results/training.png", dpi=120)

    grad_cam(model, Xte, y[te], p_te)


def grad_cam(model, X, y, p):
    # show where the network looks for a few test images
    model.eval()
    picks = list(np.where(y == 1)[0][:3]) + list(np.where(y == 0)[0][:3])
    fig, ax = plt.subplots(2, 6, figsize=(15, 5))
    for j, i in enumerate(picks):
        x = X[i:i + 1].clone().requires_grad_(True)
        feats = model.features(x)
        feats.retain_grad()
        out = model.fc(model.drop(feats.mean(dim=(2, 3)))).squeeze()
        out.backward()
        w = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * feats).sum(dim=1)).squeeze().detach().numpy()
        cam = cam / (cam.max() + 1e-8)
        img = x.detach().squeeze().numpy()
        ax[0, j].imshow(img, cmap="gray")
        ax[0, j].set_title("true %d  p=%.2f" % (y[i], p[i]))
        ax[1, j].imshow(img, cmap="gray")
        ax[1, j].imshow(np.kron(cam, np.ones((8, 8))), cmap="jet", alpha=0.4)
        ax[0, j].axis("off")
        ax[1, j].axis("off")
    plt.suptitle("Grad CAM, top row originals, bottom row where the CNN looks")
    plt.tight_layout()
    plt.savefig("results/gradcam.png", dpi=120)


if __name__ == "__main__":
    main()
