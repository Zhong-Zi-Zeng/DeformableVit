from torchvision import datasets
from PIL import Image
from models import build_model
from torchvision import transforms
from torch.utils.data import DataLoader
from util.misc import NestedTensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
import os


def get_args_parser():
    parser = argparse.ArgumentParser('DeformableVit', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--cls_num', default=100, type=int)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--enc_n_points', default=4, type=int)

    return parser


def eval(epoch):
    model.eval()
    acc = 0
    fig, ax = plt.subplots(2, 5, figsize=(16, 4))

    for i, (data, target) in enumerate(zip(test_dataset.data[:10], test_dataset.targets[:10])):
        mask = torch.zeros((1, 32, 32), device=device)
        x = trainTransform(data).to(device)
        sample = NestedTensor(x[None, ...], mask)

        output = model(sample)
        acc += torch.sum(torch.argmax(output, dim=1) == target).item()

        ax[i // 5, i % 5].axis('off')
        ax[i // 5, i % 5].set_title(
            "Pre:{}/GT:{}".format(train_dataset.classes[torch.argmax(output, dim=1)], train_dataset.classes[target]))
        ax[i // 5, i % 5].imshow(data)

    plt.savefig('./result/Epoch_{}.png'.format(epoch))
    print('Val accuracy:{:.4f}'.format(acc / 10))


def train(epoch):
    model.train()
    pbar = tqdm(range(0, len(trainLoader)), ncols=120)

    for batch, (data, target) in zip(pbar, trainLoader):
        data, target = data.to(device), target.to(device)
        mask = torch.zeros((1, 32, 32), device=device)
        sample = NestedTensor(data, mask)

        output = model(sample)
        loss = lossFunc(output, target)
        acc = torch.sum(torch.argmax(output, dim=1) == target).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description('Epoch:{} Batch:{} | loss:{:.4f} acc:{:.4f}'.format(epoch, batch, loss.item(), acc / args.batch_size))

def main(args):
    for epoch in range(args.epochs):
        train(epoch)
        eval(epoch)
        torch.save(model.state_dict(), 'weight.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeformableVit training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    trainTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.cifar.CIFAR100(root='cifar100', train=True, transform=trainTransform, download=True)
    test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=None, download=True)

    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda')
    model = build_model(args)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossFunc = torch.nn.CrossEntropyLoss()

    if os.path.isfile('./weight.pt'):
        print('Load weight')
        model.load_state_dict(torch.load('weight.pt'))

    main(args)
