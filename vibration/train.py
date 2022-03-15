from pathlib import Path

import yaml
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .dataset import Dataset
from .nets import Net

def mse(signal, pred):
    return ((signal-pred)**2).mean(dim=(1,2,3))

def auc(normal, anomaly):
    z = np.subtract.outer(anomaly, normal)
    return np.sum(z>0)/(len(normal)*len(anomaly))


def train(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    #======Read args========================================================= 
    path = Path(__file__).resolve().parents[1]
    with open(path/("configs/"+args.config+".yaml"), 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)
    with open(path/("configs/"+args.data_info+".yaml"), 'r') as f:
        data_info = yaml.load(f, yaml.FullLoader)

    model_path = path/("models/"+args.config+"_"+args.data_info)
    Path(path/"models").mkdir(parents=True, exist_ok=True)

    #======Load datasets========================================================= 
    train_dataset = Dataset(
        train=True,
        data_info=data_info['training'],
        data_path=data_info['data_path'],
        direction=cfg['direction'],
        **cfg['spectrogram']
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        shuffle=True,
        batch_size=cfg['training']['batch_size'],
        num_workers=4,
        pin_memory=True
    )

    validation_dataset = Dataset(
        train=False,
        data_info=data_info['validation'],
        data_path=data_info['data_path'],
        direction=cfg['direction'],
        **cfg['spectrogram']
    )
    validation_loader = DataLoader(
        dataset=validation_dataset, 
        shuffle=False,
        batch_size=cfg['training']['batch_size'],
        num_workers=4,
        pin_memory=True
    )

    #======Create models========================================================= 
    net = Net(
        input_size=[cfg['spectrogram']['n_fft']//2, cfg['spectrogram']['n_columns']], 
        in_channels=train_dataset.n_directions,
        **cfg['net']
    ).to(device=device, dtype=torch.float)

    if args.load_models=="true":
        net.load_state_dict(torch.load(model_path))

    net = torch.nn.DataParallel(net)


    #======Optimizers=========================================================
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=cfg['training']['lr'])
    

    #======Train=========================================================
    print('Directions: '+' and '.join(train_dataset.direction))
    for epoch in range(cfg['training']['nepochs']):
        #======Training Epoch=======
        net.train()
        total_loss = 0
        for signal, _ in train_loader:
            signal = Variable(signal).to(device=device, dtype=torch.float, non_blocking=True)
            #======Forward=======
            pred = net(signal)
            #======Losses=======
            loss = torch.mean(mse(signal, pred))
            total_loss += loss.detach()
            #======Backward=======
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            for param in net.parameters():
                param.grad = None

        #======Validation Epoch=======
        net.eval()
        loss_validation, scenarios = [], []
        for signal, scenario in validation_loader:
            signal = Variable(signal).to(device=device, dtype=torch.float)
            #======Forward=======
            pred = net(signal)
            #======Losses=======
            loss = mse(signal, pred).detach().cpu().numpy()
            loss_validation.extend(loss)
            scenarios.extend(scenario)

        #======Print log=======
        results = {scenario: \
            [loss_validation[i] for i in range(len(scenarios)) if scenarios[i]==scenario] \
                for scenario in data_info['validation']}
        # Table Header
        print()
        for _ in range((len(data_info['validation'])+1)*8*2+1): print("=", end='')
        print("\nEpoch [{}/{}]".format(epoch+1, cfg['training']['nepochs']), end='\t| ')
        for scenario in results: print((scenario+"               ")[:12], end='\t| ')
        # Training Loss
        print("\nTraining Loss", end='\t| ')
        for scenario in data_info['validation']:
            if scenario=='normales':
                print("{:.4f}".format(total_loss/len(train_loader)), end='\t\t| ')
            else:
                print("", end='\t\t\t\t| ')
        # Validation Loss
        print("\nLoss Validation", end='\t| ')
        for scenario in data_info['validation']:
            print("{:.4f}".format(np.mean(results[scenario])), end='\t\t| ')
        # AUC
        print("\nAUC Validation", end='\t| ')
        for scenario in data_info['validation']:
            if scenario=='normales':
                print("", end='\t\t\t\t| ')
            else:
                print("{:.4f}".format(
                    auc(np.array(results['normales']),np.array(results[scenario]))), end='\t\t| ')


    torch.save(net.module.state_dict(), model_path)