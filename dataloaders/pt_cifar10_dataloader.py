# import torch
# from torch.utils.data import DataLoader
# import torchvision
# import torchvision.transforms as transforms
#
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# def get_train_dataloader(batch_size):
#     train_ds = torchvision.datasets.CIFAR10(root='./datasets', train=True,
#                                             download=True, transform=transform)
#     train_dataloader = DataLoader(train_ds, batch_size=batch_size,
#                                               shuffle=True, num_workers=2)
#     return train_dataloader
#
# def get_val_dataloader(batch_size):
#     val_ds = torchvision.datasets.CIFAR10(root='./datasets', train=False,
#                                        download=True, transform=transform)
#     val_dataloader = DataLoader(val_ds, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)
#     return val_dataloader
