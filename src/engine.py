from collections import defaultdict
from tqdm.auto import tqdm
import torch

def training_loop(model: torch.nn.Module,
                  train_loader: torch.utils.data.DataLoader,
                  valid_loader: torch.utils.data.DataLoader,
                  loss_fn,
                  optimizer: torch.optim.Optimizer,
                  criterion,
                  device,
                  epochs,
                  scheduler=None) -> defaultdict(list):

    ret = defaultdict(list)
    model.to(device)
    criterion.to(device)  

    def batch_to_device(batch):
        X, y = batch
        return X.to(device), y.to(device)
    
    def train_step():
        model.train()
        train_loss, train_crit = 0, 0
        for X, y in map(batch_to_device, train_loader):
            pred = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()
            train_crit += criterion(y, pred.argmax(dim=1)).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        return train_loss / len(train_loader), train_crit / len(train_loader)
    
    def valid_step():
        model.eval()
        valid_loss, valid_crit = 0, 0
        with torch.inference_mode():
            for X, y in map(batch_to_device, valid_loader):
                pred = model(X)
                valid_loss += loss_fn(pred, y).item()
                valid_crit += criterion(y, pred.argmax(dim=1)).item()
        return valid_loss / len(valid_loader), valid_crit / len(valid_loader)

    for epoch in tqdm(range(epochs), desc=f"Running on {device}"):
        train_loss, train_crit = train_step()
        ret['train_loss'].append(train_loss)
        ret['train_crit'].append(train_crit)
        if valid_loader:
            ret['valid_loss'].append(valid_loss)
            ret['valid_crit'].append(valid_crit)

    return ret
