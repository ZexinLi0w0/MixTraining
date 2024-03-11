
import torch
import torchvision
from model_resnet import MixModel
from utils import setup_seed, WarmupCosineLR, HFDataset
import argparse
import math
from tqdm import tqdm
import wandb
import time
import tinyimagenet
from torchvision import transforms
from thop import profile
from randomaug import RandAugment
from datasets import load_dataset
import numpy as np


loss_fn = torch.nn.CrossEntropyLoss()
acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())
torch.autograd.set_detect_anomaly(True)
training_times = []
best_val_acc = 0

def train_mae_mix(train_dataloader_source, train_dataloader_target,
                  mae_mix, mae_optimizer, classifier_optimizer, unified_optimizer, lr_scheduler_mae, lr_scheduler_classifier, lr_scheduler_unified,
                  args):
    global training_times
    mae_mix.train()

    mae_losses = []
    losses = []
    acces = []
    begin_time = time.time()

    k_index = 0
    features_list = []
    labels_list = []

    if args.debug == True:
        computation_times = []


    # if args.mode == 'mae': pretrain on source dataset
    # if args.mode == 'classifier': finetune on target dataset
    # if args.mode == 'mix': mix training on both source and target dataset

    if args.mode == 'mae':
        for images, labels in tqdm(iter(train_dataloader_source)):
            # Prepare image
            images = images.to(args.device)
            labels = labels.to(args.device)
            # standard ae f(x) = x pretraining
            reconstructed_img = mae_mix(images, mode='generate')
            mae_loss = torch.mean((reconstructed_img - images) ** 2)
            mae_optimizer.zero_grad()
            mae_loss.backward()

            if args.from_pretrain == True:
                unified_optimizer.step()
                unified_optimizer.zero_grad()
            else:
                mae_optimizer.step()
                mae_optimizer.zero_grad()

            mae_losses.append(mae_loss.item())
            losses.append(0.0)
            acces.append(0.0)
            if args.debug == True:
                print(f'current mae loss is {mae_loss.item()}')
    elif args.mode == 'classifier':
        for images, labels in tqdm(iter(train_dataloader_target)):
            # Prepare image
            images = images.to(args.device)
            labels = labels.to(args.device)
            logits = mae_mix(images, mode='predict')
            classifier_loss = loss_fn(logits, labels)
            classifier_optimizer.zero_grad()
            classifier_loss.backward()
            acc = acc_fn(logits, labels)

            if args.from_pretrain == True:
                unified_optimizer.step()
                unified_optimizer.zero_grad()
            else:
                classifier_optimizer.step()
                classifier_optimizer.zero_grad()

            losses.append(classifier_loss.item())
            acces.append(acc.item())
            mae_losses.append(0.0)
            if args.debug == True:
                print(f'current classifier loss is {classifier_loss.item()}')
    elif args.mode == 'kmix':
        # mix training on both source and target dataset
        # and use the same learning rate for both mae and classifier
        # for each step, mix source and target data by 1:1, and compute loss by only one forward pass
        # then update both mae and classifier
        # Assuming train_dataloader_target and train_dataloader_source are of the same length
        for (images_target, labels_target), (images_source, labels_source) in tqdm(zip(train_dataloader_target, train_dataloader_source)):
            # handle the case when the length of train_dataloader_target and train_dataloader_source are not the same
            if len(images_target) != len(images_source):
                break


            images_target = images_target.to(args.device)
            labels_target = labels_target.to(args.device)
            images_source = images_source.to(args.device)
            labels_source = labels_source.to(args.device)

            if args.data_mix_noise == True:
                # mix with noise
                images_source = torch.randn_like(images_source).to(args.device)

            if args.data_no_mix == True:
                # first train on source dataset with reconstruction loss
                reconstructed_img = mae_mix(images_source, mode='generate')
                mae_loss = torch.mean((reconstructed_img - images_source) ** 2)
                mae_loss.backward(retain_graph=True)

                # then train on target dataset with classifier loss
                labels = labels_target
                logits = mae_mix(images_target, mode='predict')
                classifier_loss = loss_fn(logits, labels)
                classifier_loss.backward(retain_graph=True)
                acc = acc_fn(logits, labels)
            else:
                # Mix images by averaging them
                images = (1-args.lmda) * images_source + args.lmda * images_target
                # Use the target label for the mixed image
                labels = labels_target

                # Compute MAE loss
                reconstructed_img = mae_mix(images, mode='generate')
                mae_loss = torch.mean((reconstructed_img - images) ** 2)

                # Compute classifier loss
                logits = mae_mix(images, mode='predict')
                classifier_loss = loss_fn(logits, labels)
                acc = acc_fn(logits, labels)

                # Mix loss
                mix_loss = args.mix_loss_alpha * mae_loss + (1 - args.mix_loss_alpha) * classifier_loss
                mix_loss.backward()

            if args.from_pretrain == True:
                unified_optimizer.step()
                unified_optimizer.zero_grad()
            else:
                classifier_optimizer.step()
                classifier_optimizer.zero_grad()

            losses.append(classifier_loss.item())
            acces.append(acc.item())
            mae_losses.append(mae_loss.item())

    # reset dataloader
    for data in train_dataloader_target:
        pass
    for data in train_dataloader_source:
        pass

    if args.mode not in ['mae', 'classifier', 'kmix']:
        raise Exception('Unknown mode')

    if args.from_pretrain == True:
        lr_scheduler_unified.step()
    else:
        if args.mode == 'mae':
            lr_scheduler_mae.step()
        elif args.mode == 'classifier':
            lr_scheduler_classifier.step()
        elif args.mode == 'kmix':
            lr_scheduler_mae.step()
            lr_scheduler_classifier.step()
        else:
            raise Exception('Unknown mode')
    end_time = time.time()
    training_times.append(end_time - begin_time)

    # print losses
    avg_mae_loss = sum(mae_losses) / len(mae_losses)
    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(acces) / len(acces)


    if args.debug == True:
        print(f'current computation time is {sum(computation_times)}')

    return avg_mae_loss, avg_loss, avg_acc

def eval_mae_mix(val_dataloader, mae_mix, args):
    global best_val_acc
    mae_mix.eval()
    with torch.no_grad():
        losses = []
        acces = []
        for images, labels in tqdm(iter(val_dataloader)):
            images = images.to(args.device)
            labels = labels.to(args.device)
            logits = mae_mix(images, mode='predict')
            # Compute the accuracy
            loss = loss_fn(logits, labels)
            acc = acc_fn(logits, labels)
            losses.append(loss.item())
            acces.append(acc.item())
            # Log the accuracy to wandb
            # Log the accuracy to tensorboard
        avg_val_loss = sum(losses) / len(losses)
        avg_val_acc = sum(acces) / len(acces)

    # Save the model if the accuracy is the best so far
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        print(f'saving best model with acc {best_val_acc}!')
        torch.save(mae_mix, args.output_model_path)
    return avg_val_loss, avg_val_acc
def main():
    parser = argparse.ArgumentParser()
    # Add arguments for the classifier finetuning step
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256) # max batch size that can fit into RTX 6000 Ada GPU memory
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='test.pt')
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--mode', type=str, default='mix')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--mix_ratio', type=float, default=0.0)
    parser.add_argument('--mix_loss_alpha', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--load_batch_size', type=int, default=256)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--ft_epoch', type=int, default=100)
    parser.add_argument('--pt_epoch', type=int, default=100)
    parser.add_argument('--ft_warmup_epoch', type=int, default=5)
    parser.add_argument('--pt_warmup_epoch', type=int, default=20)
    parser.add_argument('--from_pretrain', action='store_true')
    parser.add_argument('--linear_probe', action='store_true')
    parser.add_argument('--info', type=str, default='')
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'resnet26', 'resnet34'])
    parser.add_argument('--lmda', type=float, default=0.5) # lambda is reserved keyword for python, use lmda instead
    parser.add_argument('--data_mix_noise', action='store_true')
    parser.add_argument('--data_no_mix', action='store_true')
    '''
        use timm.list_models() api to query: 
        avaible resnet arch right now: ['resnet18', 'resnet18d', 'resnet26', 'resnet26d', 'resnet26t', 'resnet34', 'resnet34d', 'resnet50',
          'resnet50d', 'resnet50t', 'resnet51q', 'resnet61q', 'resnet101', 'resnet101d', 'resnet152', 'resnet152d',
          'resnet200', 'resnet200d', 'resnetblur18', 'resnetblur50', 'resnetrs50', 'resnetrs101', 'resnetrs152',
          'resnetrs200', 'resnetrs270', 'resnetrs350', 'resnetrs420', 'resnetv2_50', 'resnetv2_50d', 'resnetv2_50t',
          'resnetv2_50x1_bit_distilled', 'resnetv2_50x1_bitm', 'resnetv2_50x1_bitm_in21k', 'resnetv2_50x3_bitm',
          'resnetv2_50x3_bitm_in21k', 'resnetv2_101', 'resnetv2_101d', 'resnetv2_101x1_bitm',
          'resnetv2_101x1_bitm_in21k', 'resnetv2_101x3_bitm', 'resnetv2_101x3_bitm_in21k', 'resnetv2_152',
          'resnetv2_152d', 'resnetv2_152x2_bit_teacher', 'resnetv2_152x2_bit_teacher_384', 'resnetv2_152x2_bitm',
          'resnetv2_152x2_bitm_in21k', 'resnetv2_152x4_bitm', 'resnetv2_152x4_bitm_in21k']
    '''

    args = parser.parse_args()

    setup_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_validate = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True,
                                                     transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                                   transform=transform_validate)
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_validate = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR100('data', train=True, download=True,
                                                     transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR100('data', train=False, download=True,
                                                   transform=transform_validate)
    elif args.dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])
        transform_validate = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
        ])
        train_dataset = torchvision.datasets.SVHN('data', split='train', download=True, transform=transform_train)
        val_dataset = torchvision.datasets.SVHN('data', split='test', download=True, transform=transform_validate)
    elif args.dataset == 'tinyimagenet':
        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            # Randaug as in paper https://openreview.net/forum?id=KXfjZPL5pqr
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=64, scale=(0.6, 1.0), ratio=(0.9, 1.1),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            RandAugment(n=2, m=9),

            # call AugMix with small magnitude
            # transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_validate = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_dataset = tinyimagenet.TinyImageNet('data', split='train', transform=transform_train)
        val_dataset = tinyimagenet.TinyImageNet('data', split='val', transform=transform_validate)
    elif args.dataset == 'imagenet':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            # Randaug as in paper https://openreview.net/forum?id=KXfjZPL5pqr
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=224, scale=(0.6, 1.0), ratio=(0.9, 1.1),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            RandAugment(n=2, m=9),

            # call AugMix with small magnitude
            # transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet stats
        ])
        transform_validate = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet stats
        ])

        def transform_example(example, transform):
            # Convert to PIL image and ensure RGB format
            image = example['image'].convert('RGB')

            # Apply the transform
            transformed_image = transform(image)

            # Convert back to a serializable format (numpy array)
            example['image'] = np.array(transformed_image)
            return example

        dataset = load_dataset("imagenet-1k")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        # train_dataset = train_dataset.map(lambda x: transform_example(x, transform_train), remove_columns=["image"])
        # val_dataset = val_dataset.map(lambda x: transform_example(x, transform_validate), remove_columns=["image"])

        # train_dataset = torchvision.datasets.ImageNet('data', split='train', transform=transform_train)
        # val_dataset = torchvision.datasets.ImageNet('data', split='val', transform=transform_validate)
    else:
        print(args.dataset)
        raise Exception('Unknown dataset')


    '''
        Use different data preprocessing for pretrained model from ImageNet-1K
        Resize to (224, 224) resolution instead to get much better performance
    '''
    if args.from_pretrain == True:
        if args.dataset == 'cifar10':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224), antialias=True, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_validate = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True,
                                                         transform=transform_train)
            val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                                       transform=transform_validate)
        elif args.dataset == 'cifar100':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224), antialias=True,
                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_validate = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_dataset = torchvision.datasets.CIFAR100('data', train=True, download=True,
                                                          transform=transform_train)
            val_dataset = torchvision.datasets.CIFAR100('data', train=False, download=True,
                                                        transform=transform_validate)
        elif args.dataset == 'svhn':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224), antialias=True,
                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
            ])
            transform_validate = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
            ])
            train_dataset = torchvision.datasets.CIFAR100('data', train=True, download=True,
                                                          transform=transform_train)
            val_dataset = torchvision.datasets.CIFAR100('data', train=False, download=True,
                                                        transform=transform_validate)
        elif args.dataset == 'tinyimagenet':
            transform_train = transforms.Compose([
                # Randaug as in paper https://openreview.net/forum?id=KXfjZPL5pqr
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size=224, scale=(0.6, 1.0), ratio=(0.9, 1.1),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                RandAugment(n=2, m=9),

                # call AugMix with small magnitude
                # transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            transform_validate = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            train_dataset = tinyimagenet.TinyImageNet('data', split='train', transform=transform_train)
            val_dataset = tinyimagenet.TinyImageNet('data', split='val', transform=transform_validate)
        else:
            print(args.dataset)
            raise Exception('Unknown dataset')


    if args.dataset == 'imagenet':
        train_dataset = HFDataset(train_dataset, transform_train)
        val_dataset = HFDataset(val_dataset, transform_validate)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, args.load_batch_size, shuffle=True, num_workers=32)

    # hard code source to TinyImageNet and target to CIFAR10/100 for now
    transform_train_source = transforms.Compose([
            transforms.Resize((64, 64)),
            # Randaug as in paper https://openreview.net/forum?id=KXfjZPL5pqr
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=64, scale=(0.6, 1.0), ratio=(0.9, 1.1),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            RandAugment(n=2, m=9),

            # call AugMix with small magnitude
            # transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset_source = tinyimagenet.TinyImageNet('data', split='train', transform=transform_train_source)

    train_dataset_target = None

    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # make modification to upscale to 64x64 to match the resolution of TinyImageNet
            transforms.Resize((64, 64), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_validate = transforms.Compose([
            transforms.ToTensor(),
            # make modification to upscale to 64x64 to match the resolution of TinyImageNet
            transforms.Resize((64, 64), antialias=True),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset_target = torchvision.datasets.CIFAR10('data', train=True, download=True,
                                                            transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                                   transform=transform_validate)
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # make modification to upscale to 64x64 to match the resolution of TinyImageNet
            transforms.Resize((64, 64), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_validate = transforms.Compose([
            transforms.ToTensor(),
            # make modification to upscale to 64x64 to match the resolution of TinyImageNet
            transforms.Resize((64, 64), antialias=True),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset_target = torchvision.datasets.CIFAR100('data', train=True, download=True,
                                                             transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR100('data', train=False, download=True,
                                                    transform=transform_validate)

    train_dataloader_source = torch.utils.data.DataLoader(train_dataset_source, args.load_batch_size, shuffle=True, num_workers=32)
    train_dataloader_target = torch.utils.data.DataLoader(train_dataset_target, args.load_batch_size, shuffle=True, num_workers=32)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.load_batch_size, shuffle=False, num_workers=32)

    # Model initialization
    # For standard AE pretraining, we set mask_ratio to 0.0.
    if args.pretrained_model_path is None:
        if args.dataset == 'cifar10':
            mae_mix = MixModel(backbone_name=args.model_name, num_classes=10, pretrained=args.from_pretrain).to(device)
        elif args.dataset == 'cifar100':
            mae_mix = MixModel(backbone_name=args.model_name, num_classes=100, pretrained=args.from_pretrain).to(device)
        elif args.dataset == 'svhn':
            mae_mix = MixModel(backbone_name=args.model_name, num_classes=10, pretrained=args.from_pretrain).to(device)
        elif args.dataset == 'tinyimagenet':
            mae_mix = MixModel(backbone_name=args.model_name, num_classes=200, pretrained=args.from_pretrain).to(device)
        elif args.dataset == 'imagenet':
            mae_mix = MixModel(backbone_name=args.model_name, num_classes=1000, pretrained=args.from_pretrain).to(device)
        else:
            raise Exception('Unknown dataset')

    # profile params and flops
    if args.profile:
        # import pdb; pdb.set_trace()
        import os
        import sys
        import contextlib
        # Temporarily redirect stdout to suppress [INFO] messages
        @contextlib.contextmanager
        def suppress_stdout():
            with open(os.devnull, 'w') as fnull:
                old_stdout = sys.stdout
                sys.stdout = fnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout


        input_widths = [32, 64, 224]
        forward_params = ["get_feature", "predict", "generate"]

        print("model name: ", args.model_name)
        print("=== MACs and Parameters ===")

        for input_width in input_widths:
            for forward_param in forward_params:
                print(f"Input width: {input_width}, forward param: {forward_param}")
                input = torch.randn(1, 3, input_width, input_width).to(device)
                # with suppress_stdout():
                #     macs, params = profile(mae_mix, inputs=(input, forward_param))
                macs, params = profile(mae_mix, inputs=(input, forward_param))
                macs_readable = f"{macs / 1e9:.2f} G"
                params_readable = f"{params / 1e6:.2f} M"
                print(f"MACs: {macs_readable}, Parameters: {params_readable}")
        exit(0)

    # Training loop
    training_times = []

    index = 0

    mode = []
    def _make_mode(mix_ratio, args):
        assert mix_ratio >= 0 and mix_ratio <= 1

        if args.from_pretrain == True:
            mixable_epoch = args.ft_epoch * mix_ratio
            pt_epoch_after_cal = 0
            mix_epoch_after_cal = round(mixable_epoch)
            ft_epoch_after_cal = round(args.ft_epoch - mixable_epoch)
            print(f'pt_epoch_after_cal is {pt_epoch_after_cal}, mix_epoch_after_cal is {mix_epoch_after_cal}, ft_epoch_after_cal is {ft_epoch_after_cal}')
        else:
            mixable_epoch = min(args.pt_epoch, args.ft_epoch)
            pt_epoch_after_cal = round(args.pt_epoch - mixable_epoch * mix_ratio)
            mix_epoch_after_cal = round(mixable_epoch * mix_ratio)
            ft_epoch_after_cal = round(args.ft_epoch - mixable_epoch * mix_ratio)
            print(f'pt_epoch_after_cal is {pt_epoch_after_cal}, mix_epoch_after_cal is {mix_epoch_after_cal}, ft_epoch_after_cal is {ft_epoch_after_cal}')

        mode = []
        for i in range(pt_epoch_after_cal):
            mode.append('mae')
        for i in range(mix_epoch_after_cal):
            mode.append('kmix')
        for i in range(ft_epoch_after_cal):
            mode.append('classifier')

        args.total_epoch = pt_epoch_after_cal + mix_epoch_after_cal + ft_epoch_after_cal
        print(f'args.total_epoch is {args.total_epoch}')
        print(f'len(mode) is {len(mode)}')
        return mode

    mode = _make_mode(mix_ratio=args.mix_ratio, args=args)
    assert type(args.total_epoch) == int
    if args.debug == True:
        print(f'args.total_epoch is {args.total_epoch}')
        print(f'len(mode) is {len(mode)}')
        print(f'mode is {mode}')


    # Optimizers (need to be filled in based on actual scripts)
    mae_parameter_groups = list(mae_mix.backbone.parameters()) + list(mae_mix.decoder.parameters())
    classifier_parameter_groups = list(mae_mix.backbone.parameters()) + list(mae_mix.head.parameters())

    # (Low priority) TODO: fix optmizer for mixtraining from scratch
    mae_optimizer = torch.optim.SGD(mae_parameter_groups, lr=0.1, momentum=0.9, weight_decay=5e-4)
    classifier_optimizer = torch.optim.SGD(classifier_parameter_groups, lr=0.1, momentum=0.9, weight_decay=5e-4)
    unified_optimizer = torch.optim.AdamW(list(mae_mix.head.parameters()), lr=1e-5, weight_decay=5e-4)

    if args.from_pretrain == True:
        if args.linear_probe == True:
            mae_mix.freeze_backbone_and_decoder()
            unified_optimizer = torch.optim.AdamW(list(mae_mix.head.parameters()), lr=1e-5, weight_decay=5e-4)
        else:
            if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'shvn':
                mae_optimizer = torch.optim.AdamW(mae_parameter_groups, lr=1e-5, weight_decay=5e-4)
                classifier_optimizer = torch.optim.AdamW(classifier_parameter_groups, lr=1e-5, weight_decay=5e-4)
                unified_optimizer = torch.optim.AdamW(list(mae_mix.parameters()), lr=1e-5, weight_decay=5e-4)
            else:
                # use larger learning rate for tinyimagenet since it has higher resolution
                mae_optimizer = torch.optim.AdamW(mae_parameter_groups, lr=1e-5, weight_decay=5e-4)
                classifier_optimizer = torch.optim.AdamW(classifier_parameter_groups, lr=1e-5, weight_decay=5e-4)
                unified_optimizer = torch.optim.AdamW(list(mae_mix.parameters()), lr=1e-5, weight_decay=5e-4)

    lr_scheduler_mae = WarmupCosineLR(optimizer=mae_optimizer,
                                    warmup_epochs=args.pt_warmup_epoch,
                                    total_epochs=args.pt_epoch,
                                    min_lr=1e-8)
    lr_scheduler_classifier = WarmupCosineLR(optimizer=classifier_optimizer,
                                    warmup_epochs=args.ft_warmup_epoch,
                                    total_epochs=args.ft_epoch,
                                    min_lr=1e-8)

    lr_scheduler_unified = WarmupCosineLR(optimizer=unified_optimizer,
                                        warmup_epochs=args.ft_warmup_epoch,
                                        total_epochs=args.ft_epoch,
                                        min_lr=1e-8)

    if args.enable_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="mixtrain",
            name="[{}, warmup, t-{} standard AE {}][{}mix, alpha {}, lambda {}]-{}".format(args.info, args.from_pretrain, args.model_name ,args.mix_ratio, args.mix_loss_alpha, args.lmda, args.dataset),
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.base_learning_rate,
                "architecture": "ViT-T",
                "dataset": args.dataset,
                "epochs": args.total_epoch,
            }
        )


    assert len(mode) != 0

    for epoch in range(args.total_epoch):
        args.mode = mode[index]
        index += 1
        print(f'epoch {epoch} started')
        print(f'current mode is {args.mode}')
        train_begin_time = time.time()
        avg_mae_loss, avg_loss, avg_acc = train_mae_mix(train_dataloader_source, train_dataloader_target,
                                                        mae_mix, mae_optimizer, classifier_optimizer, unified_optimizer,
                                                        lr_scheduler_mae, lr_scheduler_classifier, lr_scheduler_unified, args)
        train_end_time = time.time()
        training_times.append(train_end_time - train_begin_time)
        avg_val_loss, avg_val_acc = eval_mae_mix(val_dataloader, mae_mix, args)
        print(f'epoch {epoch}: \
            avg_mae_loss: {avg_mae_loss}, avg_loss: {avg_loss}, avg_acc: {avg_acc}, \
            avg_val_loss: {avg_val_loss}, avg_val_acc: {avg_val_acc}, training time: {sum(training_times)}')
        # log to wandb

        if args.enable_wandb:
            wandb.log({
                'mae_loss': avg_mae_loss,
                'loss': avg_loss,
                'acc': avg_acc,
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc,
                'training_time': sum(training_times),
            })

if __name__ == '__main__':
    main()