# Python 3.8.10

from model import AUTOENCODER
import torchvision.transforms as transforms
from datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

def create_plot(i, p, size):
    fig = plt.figure(figsize=(10, 7))
    rows = 2
    columns = 5

    # Original
    for x in range(i.shape[0]):
        fig.add_subplot(rows, columns, x + 1)
        img = i[x].reshape(size, size)
        plt.imshow(img)#, cmap="gray")
        plt.axis('off')

    # Reconstructed
    for x in range(p.shape[0]):
        fig.add_subplot(rows, columns, 5 + x + 1)
        img = p[x].reshape(size, size)
        plt.imshow(img)#, cmap="gray")
        plt.axis('off')
        
    return fig


def eval(model, loader, size):
    model.eval()

    for idx, (images, _) in enumerate(loader):
        images = images.reshape(-1, size*size)
        reconstructed = model(images)

        fig = create_plot(images, reconstructed.detach().numpy(), size)
        fig.savefig(f"./out.png")

        if idx == 0:
            break
        
def train(model, loader, eval_loader, size):
    # hyperparameter
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)
    num_epochs = int(args.epochs)

    # train model
    for epoch in range(num_epochs):
        model.train()
        for images, _ in loader:
            
            images = images.reshape(-1, size * size)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print (f' Epoch [{epoch+1:2}/{num_epochs}], Loss: {loss.item():.4f}')
        eval(model, eval_loader, size)

    print(' Training completed...!\n')
    torch.save(model.state_dict(), f'./model/model-identity-{args.dataset.lower()}.pth')
    print(f' Model stored at ./model/model-backdoor.pth')

def main(args):
    _SIZE = 28
    _BATCH_SIZE = 5

    # transform for images
    transform = transforms.Compose([
        transforms.Resize((_SIZE, _SIZE)),
        transforms.ToTensor()
        #transforms.Normalize((0.5, 0.5))
    ])

    # Create Model
    model = AUTOENCODER(_SIZE*_SIZE).to(args.device)

    # Optionally load model
    if args.load:
        model.load_state_dict(torch.load(f"./model/model-identity-{args.dataset.lower()}.pth", map_location=args.device))

    # Check for correct dataset
    if args.dataset.lower() != '' and args.dataset.lower() not in ['mnist']:
        print(f'\n\t[!] Error ocurred: No such dataset \"{args.dataset.lower()}\"\n')
        exit(0)

    if args.dataset.lower() == "mnist":
        dataset_path = './datasets/MNIST/'
        train_dataset = MNIST(dataset_path, train=True, transform=transform)
        eval_dataset = MNIST(dataset_path, eval=True, transform=transform)
        test_dataset = MNIST(dataset_path, test=True, transform=transform)
    
    # dataloader
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=_BATCH_SIZE)
    eval_loader = DataLoader(dataset=eval_dataset, shuffle=True, batch_size=_BATCH_SIZE)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=_BATCH_SIZE)

    print(f' Training Set: {train_dataset.__len__()}')
    print(f' Eval Set: {eval_dataset.__len__()}')
    print(f' Test Set: {test_dataset.__len__()}\n')

    if args.train:
        eval(model, eval_loader, _SIZE)
        train(model, train_loader, eval_loader, _SIZE)

    eval(model, test_loader, _SIZE)
    


if __name__ == '__main__':
    os.system("clear")
    # collect command line arguments
    parser = argparse.ArgumentParser(description='Demonstration of Backdoors in Machine Learning Models - Train ')

    parser.add_argument("--load",
                        action="store_true",
                        help="Load already trained model")

    parser.add_argument("--train",
                        action="store_true",
                        help="Train the model")
    
    parser.add_argument("--dataset",
                        required=True,
                        help="Choose between [MNIST]")
    
    parser.add_argument("--epochs",
                        default=10,
                        help="Number of training epochs")

    parser.add_argument("--device",
                        default='cpu',
                        help="Provide cuda")

    args = parser.parse_args()

    main(args)