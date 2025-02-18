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
    epochs: int = 100
    """Epochs count"""
    batch_size: int = 512
    """Batch size"""
    n_res_blocks: int = 2
    """n_res_blocks"""
    input_channels: int = 12 + 16 + 16
    """input_channels"""
    lr: float = 0.00005
    """lr"""


CUDA = torch.device('cuda')
CPU = torch.device('cpu')


def clear_and_create_dir(path):
    if path.exists():
        os.system(f'rm -rf {path}')
    os.makedirs(path)


layout = {
    "Loss": {
        "softmax_loss": ["Multiline", ["softmax_loss/train", "softmax_loss/test"]],
        "softmax_unmasked_loss": ["Multiline", ["softmax_unmasked_loss/train", "softmax_unmasked_loss/test"]],
    },
    "Metrics": {
        "all_accuracy": ["Multiline", ["all_accuracy/train", "all_accuracy/test"]],
        "alive_accuracy": ["Multiline", ["alive_accuracy/train", "alive_accuracy/test"]],
        "center_accuracy": ["Multiline", ["center_accuracy/train", "center_accuracy/test"]],
        "up_accuracy": ["Multiline", ["up_accuracy/train", "up_accuracy/test"]],
        "right_accuracy": ["Multiline", ["right_accuracy/train", "right_accuracy/test"]],
        "down_accuracy": ["Multiline", ["down_accuracy/train", "down_accuracy/test"]],
        "left_accuracy": ["Multiline", ["left_accuracy/train", "left_accuracy/test"]],
        "accuracy_team_0": ["Multiline", ["accuracy_team_0/train", "accuracy_team_0/test"]],
        "accuracy_team_1": ["Multiline", ["accuracy_team_1/train", "accuracy_team_1/test"]],
        "accuracy_match_1": ["Multiline", ["accuracy_match_1/train", "accuracy_match_1/test"]],
        "accuracy_match_2": ["Multiline", ["accuracy_match_2/train", "accuracy_match_2/test"]],
        "accuracy_match_3": ["Multiline", ["accuracy_match_3/train", "accuracy_match_3/test"]],
        "accuracy_match_4": ["Multiline", ["accuracy_match_4/train", "accuracy_match_4/test"]],
        "accuracy_match_5": ["Multiline", ["accuracy_match_5/train", "accuracy_match_5/test"]],
    }
}


def main():
    args = tyro.cli(Args)
    DATA_PATH = Path(args.data_path)
    SAVE_PATH = Path(args.save_path)

    batch_size = args.batch_size
    exp_name = f'{args.input_channels}_input_channels_{args.n_res_blocks}_res_blocks_lr_{args.lr}_bs_{batch_size}'

    EXP_DIR = SAVE_PATH / 'exps' / exp_name
    clear_and_create_dir(EXP_DIR)
    tb_writer = SummaryWriter(SAVE_PATH / 'pretrain_tb_logs' / exp_name)
    tb_writer.add_custom_scalars(layout)

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
        model_alive_mask = alive_mask.repeat((1, 5)).reshape((-1, 5, 16))
        model_out = model_out * model_alive_mask
        loss = F.cross_entropy(
            model_out,
            actions,
            reduction='none'
        )
        loss = loss * alive_mask
        loss = loss.sum(dim=1) / torch.max(alive_mask.sum(dim=1), torch.ones((actions.shape[0])).to(CUDA))
        loss = loss.mean()
        return loss
    

    def calc_unmasked_loss(model_out, actions):
        model_out = torch.Tensor(model_out)
        actions = torch.Tensor(actions).long()
        model_out = model_out.reshape(-1, 5, 16)
        loss = F.cross_entropy(
            model_out,
            actions
        )
        return loss.item()


    def calc_metrics(model_out, actions, info):
        alive_mask = get_alive_mask(info['alive_ships']).numpy()
        model_out = model_out.reshape(-1, 5, 16)
        model_actions = np.argmax(model_out, axis=1)


        is_correct = actions == model_actions
        all_accuracy = is_correct.mean()
        alive_accuracy = np.ma.masked_where(alive_mask, is_correct).mean()

        center_accuracy = np.ma.masked_where(alive_mask * (actions == 0), is_correct).mean()
        up_accuracy = np.ma.masked_where(alive_mask * (actions == 1), is_correct).mean()
        right_accuracy = np.ma.masked_where(alive_mask * (actions == 2), is_correct).mean()
        down_accuracy = np.ma.masked_where(alive_mask * (actions == 3), is_correct).mean()
        left_accuracy = np.ma.masked_where(alive_mask * (actions == 4), is_correct).mean()

        players = info['player'].numpy()
        team_0_mask = np.repeat(players == 0, 16).reshape((-1, 16))
        team_1_mask = np.repeat(players == 1, 16).reshape((-1, 16))
        accuracy_team_0 = np.ma.masked_where(alive_mask * team_0_mask, is_correct).mean()
        accuracy_team_1 = np.ma.masked_where(alive_mask * team_1_mask, is_correct).mean()

        step = info['step'].numpy()
        match = step // (100 + 1) + 1
        match_1_mask = np.repeat(match == 1, 16).reshape((-1, 16))
        match_2_mask = np.repeat(match == 2, 16).reshape((-1, 16))
        match_3_mask = np.repeat(match == 3, 16).reshape((-1, 16))
        match_4_mask = np.repeat(match == 4, 16).reshape((-1, 16))
        match_5_mask = np.repeat(match == 5, 16).reshape((-1, 16))
        accuracy_match_1 = np.ma.masked_where(alive_mask * match_1_mask, is_correct).mean()
        accuracy_match_2 = np.ma.masked_where(alive_mask * match_2_mask, is_correct).mean()
        accuracy_match_3 = np.ma.masked_where(alive_mask * match_3_mask, is_correct).mean()
        accuracy_match_4 = np.ma.masked_where(alive_mask * match_4_mask, is_correct).mean()
        accuracy_match_5 = np.ma.masked_where(alive_mask * match_5_mask, is_correct).mean()

        return {
            'softmax_unmasked_loss': calc_unmasked_loss(model_out, actions),
            'all_accuracy': all_accuracy,
            'alive_accuracy': alive_accuracy,
            'center_accuracy': center_accuracy,
            'up_accuracy': up_accuracy,
            'right_accuracy': right_accuracy,
            'down_accuracy': down_accuracy,
            'left_accuracy': left_accuracy,
            'accuracy_team_0': accuracy_team_0,
            'accuracy_team_1': accuracy_team_1,
            'accuracy_match_1': accuracy_match_1,
            'accuracy_match_2': accuracy_match_2,
            'accuracy_match_3': accuracy_match_3,
            'accuracy_match_4': accuracy_match_4,
            'accuracy_match_5': accuracy_match_5,
        }


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

            model_out = model_out.cpu().detach().numpy()
            actions = actions.cpu().detach().numpy()
            metrics = calc_metrics(model_out, actions, batch['info'])
            for metric, value in metrics.items():
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
            
            model_out = model_out.cpu().detach().numpy()
            actions = actions.cpu().detach().numpy()
            metrics = calc_metrics(model_out, actions, batch['info'])
            for metric, value in metrics.items():
                test_metrics[metric] += value * obs.size(0)

            test_loss += loss.item() * obs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)
        for metric in train_metrics.keys():
            train_metrics[metric] /= len(train_loader.dataset)
            test_metrics[metric] /= len(test_loader.dataset)

        tb_writer.add_scalar('softmax_loss/train', train_loss, epoch + 1)
        tb_writer.add_scalar('softmax_loss/test', test_loss, epoch + 1)
        for metric in train_metrics.keys():
            tb_writer.add_scalar(f'{metric}/train', train_metrics[metric], epoch + 1)
            tb_writer.add_scalar(f'{metric}/test', test_metrics[metric], epoch + 1)
        tb_writer.flush()
    
    model_save_path = EXP_DIR / 'model.pt'
    print(f'Saving model to {model_save_path}')
    model = model.to(CPU).eval()
    torch.jit.script(model).save(model_save_path)


if __name__ == "__main__":
    main()