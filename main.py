# -*- coding: utf-8 -*-
import time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import gpytorch

def seed_all(seed=1234):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ResDenseBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj = None if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim)
        nn.init.normal_(self.fc1.weight, 0.0, np.sqrt(1.0 / in_dim)); nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, 0.0, np.sqrt(1.0 / hidden_dim)); nn.init.zeros_(self.fc2.bias)
        if self.proj is not None:
            nn.init.normal_(self.proj.weight, 0.0, np.sqrt(1.0 / in_dim)); nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        h = F.selu(self.fc1(x)); h = self.fc2(h)
        return (x if self.proj is None else self.proj(x)) + h

class ResMLP(nn.Module):
    def __init__(self, depth=11, in_dim=37, hidden_dim=20):
        super().__init__()
        self.block0 = ResDenseBlock(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResDenseBlock(hidden_dim, hidden_dim) for _ in range(depth)])
        self.in_skip = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.in_skip.weight, 0.0, np.sqrt(1.0 / in_dim)); nn.init.zeros_(self.in_skip.bias)
        nn.init.normal_(self.out.weight, 0.0, np.sqrt(1.0 / hidden_dim)); nn.init.zeros_(self.out.bias)
    def forward(self, x):
        h = self.block0(x)
        for blk in self.blocks: h = blk(h)
        return self.out(h + self.in_skip(x))

def mape_loss(y_pred, y_true, eps=1e-8):
    denom = torch.clamp(torch.abs(y_true), min=eps)
    return torch.mean(torch.abs((y_true - y_pred) / denom)) * 100.0

def mape_np(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).reshape(-1); y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, input_dimension, output_dimension):
        q = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        s = gpytorch.variational.VariationalStrategy(self, inducing_points, q, learn_inducing_locations=True)
        super().__init__(s)
        self.mean_module = gpytorch.means.ZeroMean()
        self.k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=tuple(range(input_dimension))))
        self.k2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=tuple(range(input_dimension, input_dimension + output_dimension))))
        self.covar_module = self.k1 + self.k2
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

def plot_90pi_5days(y_true, y_mean, y_var, points_per_day=24, start=0, title="5-day forecast with 90% prediction interval",
                    save_png="pi90_5days.png", save_pdf="pi90_5days.pdf"):
    y_true = np.asarray(y_true).reshape(-1)
    y_mean = np.asarray(y_mean).reshape(-1)
    y_var  = np.asarray(y_var).reshape(-1)
    n = points_per_day * 5
    end = min(start + n, len(y_true), len(y_mean), len(y_var))
    yt = y_true[start:end]; mu = y_mean[start:end]; var = np.maximum(y_var[start:end], 0.0)
    sigma = np.sqrt(var)
    z = 1.6448536269514722
    lo = mu - z * sigma
    hi = mu + z * sigma
    x = np.arange(end - start)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.fill_between(x, lo, hi, alpha=0.20, linewidth=0.0, label="90% PI")
    ax.plot(x, mu, linewidth=1.5, label="Prediction (mean)")
    ax.plot(x, yt, linewidth=1.2, linestyle="--", label="Ground truth")

    ticks = np.arange(0, end - start + 1, points_per_day)
    ax.set_xticks(ticks[:-1])
    ax.set_xticklabels([f"Day {i+1}" for i in range(len(ticks[:-1]))])

    ax.set_xlabel("Time"); ax.set_ylabel("Load"); ax.set_title(title, pad=6)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=3, handlelength=2.0, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(save_png, dpi=600, bbox_inches="tight")
    fig.savefig(save_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_png, "and", save_pdf)

if __name__ == "__main__":
    seed_all(1234)
    DATA_CSV = "Chicago_data_all.csv"
    MOB_CSV  = "Chicago_mobility_all.csv"

    df = pd.read_csv(DATA_CSV)
    df1 = pd.read_csv(MOB_CSV)

    data  = df.values.astype(np.float32)
    data1 = df1.values.astype(np.float32)

    n, m = data.shape
    n1, m1 = data1.shape

    max_value  = np.max(np.abs(data[:, :]), axis=0)
    max_load   = np.max(data[:, 0])

    for i in range(m):
        data[:, i]  = data[:, i]  / (max_value[i] + 1e-12)
        data1[:, i] = data1[:, i] / (max_value[i] + 1e-12)

    for i in range(m, m1):
        denom = np.max(np.abs(data1[:, i])) + 1e-12
        data1[:, i] = data1[:, i] / denom

    train_size, val = 20000, 500
    train_X = data[:train_size, 1:]
    train_Y = data[:train_size, 0]
    test_X  = data[train_size:train_size + val, 1:]
    test_Y  = data[train_size:train_size + val, 0]

    s = -24 * 30
    n_win = 72
    m_idx = s + n_win
    j_idx = m_idx + 24 * 3
    train_X_M = data1[s:s + n_win, 1:38]
    train_Y_M = data1[s:s + n_win, 0]
    test_X_M  = data1[m_idx:j_idx, 1:38]
    test_Y_M  = data1[m_idx:j_idx, 0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = torch.from_numpy(train_X).to(device)
    Ytr = torch.from_numpy(train_Y).to(device).view(-1, 1)
    Xte = torch.from_numpy(test_X).to(device)
    Yte = torch.from_numpy(test_Y).to(device).view(-1, 1)

    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=128, shuffle=True, drop_last=False)
    model = ResMLP(depth=11, in_dim=37, hidden_dim=20).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    t1 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optim.zero_grad(set_to_none=True)
            loss = mape_loss(model(xb), yb)
            loss.backward()
            optim.step()
        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                val_mape = mape_loss(model(Xte), Yte).item()
            print(f"Epoch {ep:03d} | val_mape={val_mape:.4f}")
    t2 = time.time()

    XtrM = torch.from_numpy(train_X_M).to(device)
    XteM = torch.from_numpy(test_X_M).to(device)
    model.eval()
    with torch.no_grad():
        train_pred = model(XtrM).detach().cpu().numpy().reshape(-1, 1)
        test_pred  = model(XteM).detach().cpu().numpy().reshape(-1, 1)
    train_NN_errors = (train_Y_M.reshape(-1, 1) - train_pred).astype(np.float32)

    mob_feat_train = data1[s:s + n_win, 1:].astype(np.float32)
    mob_feat_test  = data1[m_idx:j_idx, 1:].astype(np.float32)
    combined_train_data = np.concatenate([mob_feat_train, train_pred.astype(np.float32)], axis=1)
    combined_test_data  = np.concatenate([mob_feat_test,  test_pred.astype(np.float32)], axis=1)

    Xgp_train = torch.from_numpy(combined_train_data).to(device)
    Ygp_train = torch.from_numpy(train_NN_errors).to(device).view(-1)
    Xgp_test  = torch.from_numpy(combined_test_data).to(device)

    input_dimension = mob_feat_train.shape[1]
    output_dimension = 1
    Z = Xgp_train[:20, :].clone()

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    gp_model = VariationalGPModel(Z, input_dimension, output_dimension).to(device)

    gp_model.train(); likelihood.train()
    mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=Xgp_train.size(0))
    gp_optim = torch.optim.Adam([{"params": gp_model.parameters()}, {"params": likelihood.parameters()}], lr=1e-2)

    gp_epochs = 800
    t3 = time.time()
    for it in range(1, gp_epochs + 1):
        gp_optim.zero_grad(set_to_none=True)
        loss = -mll(gp_model(Xgp_train), Ygp_train)
        loss.backward()
        gp_optim.step()
        if it % 200 == 0 or it == 1:
            print(f"GP iter {it:04d} | negELBO={loss.item():.4f}")
    t4 = time.time()

    gp_model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(gp_model(Xgp_test))
        mean = pred_dist.mean.detach().cpu().numpy().reshape(-1, 1)
        var  = pred_dist.variance.detach().cpu().numpy().reshape(-1, 1)

    test_final_predictions = test_pred.reshape(-1) + mean.reshape(-1)
    MAPE = mape_np(test_Y_M.reshape(-1), test_final_predictions)
    print(f"test mape after GP: {MAPE:.4f}")

    mean_ = test_final_predictions * max_load
    var_  = var.reshape(-1) * (max_load ** 2)

    y_true_ = test_Y_M.reshape(-1) * max_load
    plot_90pi_5days(y_true_, mean_, var_, points_per_day=24, start=0)
