# fit_CNN_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import glob
import os

# 1. 定义数据集
class SimulationDataset(Dataset):
    def __init__(self, file_list, window_size):
        self.samples = []
        for file in file_list:
            with open(file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            for sim in data['Results']['listOfSingleSimulationDicts']:
                X_ex = np.array(list(sim['exInputSpikeTimes'].values()))
                X_inh = np.array(list(sim['inhInputSpikeTimes'].values()))
                X = np.concatenate([X_ex, X_inh], axis=0)  # (num_segments, sim_duration)
                y_spike = sim['outputSpikeTimes'].astype(float)
                y_soma = sim['somaVoltageLowRes']
                # 这里假设每个样本是一个窗口
                for t in range(window_size, X.shape[1]):
                    x_win = X[:, t-window_size:t]
                    y_spk_win = np.zeros(window_size)
                    y_spk_win[(y_spike[(y_spike >= t-window_size) & (y_spike < t)] - (t-window_size)).astype(int)] = 1
                    y_soma_win = y_soma[t-window_size:t]
                    self.samples.append((x_win, y_spk_win, y_soma_win))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y_spk, y_soma = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32).T, torch.tensor(y_spk, dtype=torch.float32), torch.tensor(y_soma, dtype=torch.float32)

# 2. 定义CNN模型
class SimpleTCN(nn.Module):
    def __init__(self, input_channels, network_depth, num_filters_per_layer, filter_sizes_per_layer):
        super().__init__()
        layers = []
        in_ch = input_channels
        for i in range(network_depth):
            layers.append(nn.Conv1d(in_ch, num_filters_per_layer, filter_sizes_per_layer[i], padding='same'))
            layers.append(nn.ReLU())
            in_ch = num_filters_per_layer
        self.conv = nn.Sequential(*layers)
        self.spike_head = nn.Conv1d(in_ch, 1, 1)
        self.soma_head = nn.Conv1d(in_ch, 1, 1)
    def forward(self, x):
        # x: (batch, time, channels) -> (batch, channels, time)
        x = x.permute(0, 2, 1)
        feat = self.conv(x)
        spike_out = torch.sigmoid(self.spike_head(feat)).squeeze(1)
        soma_out = self.soma_head(feat).squeeze(1)
        return spike_out, soma_out

# 3. 训练主流程
def train_and_save_pytorch(network_depth, num_filters_per_layer, input_window_size, num_epochs, models_dir):
    # 数据准备
    train_files = glob.glob('./Single_Neuron_InOut_SJC/data/L5PC_NMDA_train/*.p')
    valid_files = glob.glob('./Single_Neuron_InOut_SJC/data/L5PC_NMDA_valid/*.p')
    test_files  = glob.glob('./Single_Neuron_InOut_SJC/data/L5PC_NMDA_test/*.p')
    train_dataset = SimulationDataset(train_files, input_window_size)
    valid_dataset = SimulationDataset(valid_files, input_window_size)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=2)

    # 网络结构
    num_segments = train_dataset[0][0].shape[1]
    filter_sizes_per_layer = [54] + [24] * (network_depth - 1)
    model = SimpleTCN(num_segments, network_depth, num_filters_per_layer, filter_sizes_per_layer)
    model = model.cuda() if torch.cuda.is_available() else model

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn_spike = nn.BCELoss()
    loss_fn_soma = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y_spk, y_soma in train_loader:
            x, y_spk, y_soma = x.cuda(), y_spk.cuda(), y_soma.cuda() if torch.cuda.is_available() else (x, y_spk, y_soma)
            optimizer.zero_grad()
            pred_spk, pred_soma = model(x)
            loss = loss_fn_spike(pred_spk, y_spk) + 0.02 * loss_fn_soma(pred_soma, y_soma)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} train loss: {train_loss/len(train_loader):.4f}")

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y_spk, y_soma in valid_loader:
                x, y_spk, y_soma = x.cuda(), y_spk.cuda(), y_soma.cuda() if torch.cuda.is_available() else (x, y_spk, y_soma)
                pred_spk, pred_soma = model(x)
                loss = loss_fn_spike(pred_spk, y_spk) + 0.02 * loss_fn_soma(pred_soma, y_soma)
                val_loss += loss.item()
        val_loss /= len(valid_loader)
        print(f"Epoch {epoch+1}/{num_epochs} val loss: {val_loss:.4f}")

        # 保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(models_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(models_dir, f"best_model_epoch{epoch+1}.pt"))
            print(f"Saved best model at epoch {epoch+1}")
