import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import random
from torch import optim, nn

from train_test.train import train
from train_test.test import test

parser = argparse.ArgumentParser(description='Sparse Double Descent in Vision Transformers: real or phantom threat?')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
parser.add_argument('--num_classes', type=int, default='10', help='10 for CIFAR-10; 100 for CIFAR-100')
parser.add_argument('--model', default='vit', help='vit, swint or resnet-18')
parser.add_argument('--batch_size', type=int, default='512')
parser.add_argument('--epochs', type=int, default='200')
parser.add_argument('--weight_decay', type=float, default='0.03')
parser.add_argument('--amount_noise', type=float, default='0.1')

args = parser.parse_args()

## Using GPU
GPU_id = torch.cuda.current_device()
cuda = "cuda:"+str(GPU_id)
device = torch.device(cuda)

## Preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

## CIFAR Dataset
if args.num_classes ==100:
    train_dataset = torchvision.datasets.CIFAR100(root='~/data/cifar-100/',
                                                train=True, 
                                                transform=transform,
                                                download=True)

    test_dataset = torchvision.datasets.CIFAR100(root='~/data/cifar-100/',
                                                train=False, 
                                                transform=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                                                download=True)
else :
    train_dataset = torchvision.datasets.CIFAR10(root='~/data/cifar-10/',
                                                train=True, 
                                                transform=transform,
                                                download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='~/data/cifar-10/',
                                                train=False, 
                                                transform=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                                                download=True)

## Noise injection :
for i in range(int(len(train_dataset)*args.amount_noise)):
    label = train_dataset.targets[i]
    a=random.randint(0,args.num_classes-1)
    while label == a:
        a=random.randint(0,args.num_classes-1)
    train_dataset.targets[i] = a

## Dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size, 
                                        shuffle=True,
                                        num_workers=32)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=args.batch_size, 
                                        shuffle=False,
                                        num_workers=32)

## Load model
if args.model == "vit":
    from models.vit import ViT
    model = ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = args.num_classes,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
        ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

elif args.model == "swin":
    from models.swin import swin_t
    model = swin_t(window_size= 4,
               num_classes=args.num_classes,
               downscaling_factors=(2,2,2,1)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

elif args.model == "resnet-18":
    from models.resnet import ResNet
    model = ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)


## Loss, optimizer and scheduler:
loss_fn=nn.CrossEntropyLoss().to(device)

## Training and Evalutations :
train_losses=[]
train_accu=[]

eval_losses=[]
eval_accu=[]
iteration = 0

for epoch in range(0,args.epochs):
    train_acc, train_loss = train(model, epoch, train_loader, device, loss_fn, optimizer, train_accu, train_losses)
    test_acc, test_loss = test(model, test_loader, device, loss_fn, eval_accu, eval_losses)
    scheduler.step()

torch.save(model,"CIFAR-"+str(args.num_classes)+"_"+args.model)

## Pruning loop :
for i in range(1, 35):
    sparsity = 1-(1-0.2)**i
    
    if args.model == "vit" or args.model == "swin":
        layers_to_prune = [(module, "weight") for module in filter(lambda m: type(m) == torch.nn.Linear, model.modules())]
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        layers_to_prune = [(module, "weight") for module in filter(lambda m: type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear, model.modules())]
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    torch.nn.utils.prune.global_unstructured(layers_to_prune, pruning_method=torch.nn.utils.prune.L1Unstructured, amount=0.2)

    loss_fn=nn.CrossEntropyLoss().to(device)

    train_losses=[]
    train_accu=[]
    eval_losses=[]
    eval_accu=[]

    for epoch in range(0,args.epochs):
        train_acc, train_loss = train(model, epoch, train_loader, device, loss_fn, optimizer, train_accu, train_losses)
        test_acc, test_loss = test(model, test_loader, device, loss_fn, eval_accu, eval_losses)
        scheduler.step()
    
    torch.save(model, "CIFAR-"+str(args.num_classes)+"_"+args.model+"_pruned_"+str(sparsity))
