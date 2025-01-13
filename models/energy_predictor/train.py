import tyro
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataset import EnergyDataset
import torchvision.utils as vutils
import numpy as np
from torch.utils.tensorboard import SummaryWriter



@dataclass
class Args:
    data_path: str = '/home/artemveshkin/dev/luxai-s3/energy_logs'
    """Data path (energy logs)"""
    save_path: str = '/home/artemveshkin/dev/luxai-s3/models/energy_predictor'
    """Checkpoints and logs save path"""
    epochs: int = 30
    """Epochs count"""
    batch_size: int = 512
    """Batch size"""
    unet_in_channels: int = 8
    """UNet in channels"""
    unet_init_features: int = 32
    """UNet init_features"""


CUDA = torch.device('cuda')
CPU = torch.device('cpu')


def clear_and_create_dir(path):
    if path.exists():
        os.system(f'rm -rf {path}')
    os.makedirs(path)


def main():
    args = tyro.cli(Args)
    DATA_PATH = Path(args.data_path)
    SAVE_PATH = Path(args.save_path)

    batch_size = args.batch_size
    exp_name = f'brain_unet_bs_{batch_size}_init_features_{args.unet_init_features}'

    EXP_DIR = SAVE_PATH / 'exps' / exp_name
    clear_and_create_dir(EXP_DIR)
    tb_writer = SummaryWriter(SAVE_PATH / 'tb_logs' / exp_name)
    

    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 
        'unet', 
        in_channels=args.unet_in_channels, 
        out_channels=1,
        init_features=args.unet_init_features, 
        pretrained=False
    )
    model = nn.Sequential(
        model,
        nn.AvgPool2d(2)
    )
    model.to(CUDA)
    summary(model, input_size=(args.unet_in_channels, 48, 48))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    train_loader = DataLoader(EnergyDataset(DATA_PATH / 'train'), batch_size=batch_size, shuffle=True)
    test_dataset = EnergyDataset(DATA_PATH / 'test')

    eval_x, eval_x_in, eval_gt = [], [], []
    for eval_example_idx in [6, 7, 9, 10, 11, 12]:
        eval_x.append(torch.unsqueeze(torch.Tensor(test_dataset[eval_example_idx]['original_x'][0, :, :]), 0))
        eval_x_in.append(test_dataset[eval_example_idx]['upsampled_x'])
        eval_gt.append(torch.unsqueeze(torch.Tensor(test_dataset[eval_example_idx]['gt'][0, :, :]), 0))
    eval_x_in = torch.Tensor(np.array(eval_x_in)).to(CUDA)


    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        train_loss = 0.0
        test_loss = 0.0

        print('Train step')
        model.train()
        for batch in tqdm(train_loader):
            x = batch['upsampled_x'].to(CUDA)
            gt = batch['gt'].to(CUDA)

            optimizer.zero_grad()
            model_out = model(x)
            loss = criterion(model_out, gt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)

        print('Eval step')
        model.eval()

        nrows = len(eval_x)
        grid_eval_x = vutils.make_grid(eval_x, nrow=nrows)
        eval_model_out = model(eval_x_in).to(CPU)
        eval_model_out = torch.clamp(eval_model_out, 0., 1.)
        grid_model_out = vutils.make_grid(eval_model_out, nrow=nrows)
        grid_eval_gt = vutils.make_grid(eval_gt, nrow=nrows)
        grid = torch.cat((
            grid_eval_x,
            grid_model_out,
            grid_eval_gt
        ), 1)
        tb_writer.add_image('Energy predictions', grid, epoch + 1)

        for batch in tqdm(test_loader):
            x = batch['upsampled_x'].to(CUDA)
            gt = batch['gt'].to(CUDA)
            
            model_out = model(x)
            loss = criterion(model_out, gt)
            
            test_loss += loss.item() * x.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)

        tb_writer.add_scalars(
            'MSE_loss',
            {
                'train': train_loss,
                'test': test_loss
            },
            epoch + 1
        )

        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tTest Loss: {test_loss:.6f}')

    model_save_path = EXP_DIR / 'model.pt'
    print(f'Saving model to {model_save_path}')
    model = model.to(CPU).eval()
    torch.jit.script(model).save(model_save_path)


if __name__ == "__main__":
    main()