import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import PlayerCNN
from src.preprocessing import prepare_targets
from src.utils import set_seed


def train_KFolds(meta_df, feature_df, batch_size=1024, seed=42, n_folds=5, debug=False):
    oof = np.zeros(len(meta_df))
    criterion = nn.BCELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    # split 2017 and 2018 data temporarily for CV setup
    idx_2017 = meta_df.index[meta_df.Season == 2017]
    idx_2018 = meta_df.index[meta_df.Season == 2018]
    meta_df_2017, feature_df_2017 = meta_df.loc[idx_2017], feature_df[idx_2017]
    target_2017 = meta_df_2017.pop('Yards').values
    meta_df_2018, feature_df_2018 = meta_df.loc[idx_2018], feature_df[idx_2018]
    target_2018 = meta_df_2018.pop('Yards').values

    gkf = GroupKFold(n_splits=n_folds)
    games = meta_df_2018.GameId

    for fold, (train_idx, val_idx) in enumerate(gkf.split(feature_df_2018, target_2018, groups=games)):
        if debug:
            if fold != 1:
                continue

        # split 2018 data into training and evaluation set
        xtrain_2018, ytrain_2018 = feature_df_2018[train_idx], target_2018[train_idx]
        xval, yval = feature_df_2018[val_idx], target_2018[val_idx]
        yval = prepare_targets(yval)

        # append 2017 data and 2018 training data --> full train set
        xtrain = np.vstack((feature_df_2017, xtrain_2018))
        ytrain = np.hstack((target_2017, ytrain_2018))
        ytrain = prepare_targets(ytrain)

        # create torch tensors
        xtrain, ytrain = torch.from_numpy(xtrain), torch.from_numpy(ytrain)
        xval, yval = torch.from_numpy(xval), torch.from_numpy(yval)

        train_set = TensorDataset(xtrain, ytrain)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_set = TensorDataset(xval, yval)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        model = PlayerCNN()
        optimizer = optim.Adam(model.parameters(), lr=.001)

        summary_path = 'runs/' + time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(summary_path)

        for epoch in range(1, 50):
            trn_loss = 0.0
            val_loss = 0.0

            for i, (features, targets) in enumerate(train_loader):
                features, targets = features.float().to(device), targets.float().to(device)
                writer.add_graph(model, features)

                model.train()
                outputs = model(features)
                loss = criterion(outputs, targets)
                trn_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                print('[%d] loss: %.5f' % (epoch, trn_loss / len(train_loader)))
                writer.add_scalar('training loss', trn_loss / len(train_loader), epoch)

                for i, (features, targets) in enumerate(val_loader):
                    features, targets = features.float().to(device), targets.float().to(device)

                    model.eval()
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    if i == 0:
                        eval_set = outputs.cpu().numpy()
                    else:
                        eval_set = np.vstack((eval_set, outputs.cpu().numpy()))

                print('[%d] validation loss: %.5f' % (epoch, val_loss / len(val_loader)))
                writer.add_scalar('validation loss', val_loss / len(val_loader), epoch)

                yval_cum = np.clip(np.cumsum(yval.numpy(), axis=1), 0, 1)
                eval_set = np.clip(np.cumsum(eval_set, axis=1), 0, 1)
                validation_score = ((eval_set - yval_cum) ** 2).sum(axis=1).sum(axis=0) / (199 * yval_cum.shape[0])
                print('[%d] validation score: %.5f' % (epoch, validation_score))
                writer.add_scalar('validation score', validation_score, epoch)
        writer.close()
