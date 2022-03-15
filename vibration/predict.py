from pathlib import Path

import yaml
import h5py

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .dataset import Dataset
from .nets import Net

def mse(signal, pred):
    return ((signal-pred)**2).mean(dim=(1,2))

def predict(args):

    device = torch.device('cpu')

    #======Read args========================================================= 
    path = Path(__file__).resolve().parents[1]
    with open(path/("configs/"+args.config+".yaml"), 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)
    with open(path/("configs/"+args.data_info+".yaml"), 'r') as f:
        data_info = yaml.load(f, yaml.FullLoader)

    model_path = path/("models/"+args.config+"_"+args.data_info)
    pred_path = path/("predictions/"+args.config+"_"+args.data_info)
    Path(path/"predictions").mkdir(parents=True, exist_ok=True)


    #======Load datasets========================================================= 
    dataset = Dataset(
        train=False,
        data_info=data_info['evaluation'],
        data_path=data_info['data_path'],
        direction=cfg['direction'],
        **cfg['spectrogram']
    )
    loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1024)


    #======Create models========================================================= 
    net = Net(in_channels=len(cfg['direction']),*cfg['net']).\
        to(device=device, dtype=torch.float).eval()

    net.load_state_dict(torch.load(model_path, map_location=device))


    #======Make predictions========================================================= 
    loss_list, scenarios = [], []
    for signal, scenario in loader:
        signal = Variable(signal).to(device=device, dtype=torch.float)
        #======Forward=======
        pred = net(signal)
        #======Losses=======
        loss = torch.mean(mse(signal, pred)).detach().cpu().numpy()
        loss_list.extend(loss)
        scenarios.extend(scenario.numpy())

    results = {scenario: \
        [loss_list[i] for i in range(len(scenarios)) if scenarios[i]==scenario] \
            for scenario in data_info['evaluation']}


    #======Save predictions========================================================= 
    with h5py.File(pred_path, 'w') as f:
        f.create_dataset("results", data=results)