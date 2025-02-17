import tyro
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torchsummary import summary
from tqdm.auto import tqdm
import numpy as np
from states_dataset import StatesDataset
from torch.utils.tensorboard import SummaryWriter
from ppo_policy import ActorNet
import torch.nn.functional as F
from collections import defaultdict



@dataclass
class Args:
    data_path: str = '/home/artemveshkin/dev/luxai-s3/state_logs'
    """Data path (state logs)"""
    save_path: str = '/home/artemveshkin/dev/luxai-s3/agents/ppo_pretrained_1'
    """Checkpoints and logs save path"""
    epochs: int = 30
    """Epochs count"""
    batch_size: int = 512
    """Batch size"""
    n_res_blocks: int = 2
    """n_res_blocks"""
    input_channels: int = 5 + 16 + 16
    """input_channels"""
    lr: float = 0.00005
    """lr"""


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
    exp_name = f'{args.input_channels}_input_channels_{args.n_res_blocks}_res_blocks_lr_{args.lr}_bs_{batch_size}'

    EXP_DIR = SAVE_PATH / 'exps' / exp_name
    clear_and_create_dir(EXP_DIR)
    tb_writer = SummaryWriter(SAVE_PATH / 'pretrain_tb_logs' / exp_name)

    model_params = {
        'input_channels': args.input_channels,
        'n_res_blocks': args.n_res_blocks,
        'all_channel': args.input_channels * 2,
        'n_actions': 5
    }
    model = ActorNet(model_params)
    model.to(CUDA)
    summary(model, input_size=(model_params['input_channels'], 24, 24))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        StatesDataset(DATA_PATH / 'train'),
        batch_size=batch_size,
        shuffle=True,
        num_workers=20
    )
    test_loader = DataLoader(
        StatesDataset(DATA_PATH / 'test'),
        batch_size=batch_size,
        shuffle=False,
        num_workers=20
    )

    
    def get_alive_mask(alive_ships):
        mask = np.zeros((len(alive_ships), 16))
        for row_idx, row in enumerate(alive_ships):
            if len(row) > 0:
                alive_ships_idxs = list(map(int, row.split(',')))
                mask[row_idx, alive_ships_idxs] = 1.
        return torch.Tensor(mask)


    def calc_loss(model_out, actions, alive_ships):
        model_out = model_out.reshape(-1, 5, 16)
        alive_mask = get_alive_mask(alive_ships).to(CUDA)
        loss = F.cross_entropy(
            model_out,
            actions,
            reduction='none'
        )
        loss = loss * alive_mask
        loss = loss.mean()
        return loss
    

    def calc_metrics(model_out, actions, info):
        alive_mask = get_alive_mask(alive_ships).to(CUDA)
        model_out = model_out.reshape(-1, 5, 16)
        model_actions = torch.argmax(model_out, dim=1)
        print(f'model_actions.shape={model_actions.shape}')
        print(f'model_out[0]={model_out[0]}')
        print(f'model_actions[0]={model_actions[0]}')

        return {'123': 1.}


    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        train_loss = 0.0
        test_loss = 0.0

        train_metrics = defaultdict(float)
        test_metrics = defaultdict(float)

        print('Train step')
        model.train()
        for batch in tqdm(train_loader):
            obs = batch['obs'].to(CUDA)
            actions = batch['actions'].to(CUDA)
            alive_ships = batch['info']['alive_ships']

            optimizer.zero_grad()
            model_out = model(obs)
            
            loss = calc_loss(model_out, actions, alive_ships)
            loss.backward()
            optimizer.step()

            for metric, value in calc_metrics(model_out, actions, batch['info']):
                train_metrics[metric] += value * obs.size(0)

            train_loss += loss.item() * obs.size(0)

        print('Eval step')
        model.eval()

        for batch in tqdm(test_loader):
            obs = batch['obs'].to(CUDA)
            actions = batch['actions'].to(CUDA)
            alive_ships = batch['info']['alive_ships']

            model_out = model(obs)
            loss = calc_loss(model_out, actions, alive_ships)
            
            for metric, value in calc_metrics(model_out, actions, batch['info']):
                test_metrics[metric] += value * obs.size(0)

            test_loss += loss.item() * obs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)
        for metric in train_metrics.keys():
            train_metrics[metric] /= len(train_loader.dataset)
            test_metrics[metric] /= len(test_loader.dataset)

        tb_writer.add_scalars(
            'softmax_loss',
            {
                'train': train_loss,
                'test': test_loss
            },
            epoch + 1
        )
        for metric in train_metrics.keys():
            tb_writer.add_scalars(
                metric,
                {
                    'train': train_metrics[metric],
                    'test': test_metrics[metric]
                },
                epoch + 1
            )
    
    model_save_path = EXP_DIR / 'model.pt'
    print(f'Saving model to {model_save_path}')
    model = model.to(CPU).eval()
    torch.jit.script(model).save(model_save_path)


if __name__ == "__main__":
    main()