import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from model_vit import MAE_Encoder, MAE_Mix  # Assuming that MAE_Decoder is also defined in model_vit.py
from utils import setup_seed  # Assuming this is a function to set the random seed
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
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Convert to PIL image and ensure RGB format
        # image = Image.fromarray(self.hf_dataset[idx]['image']).convert('RGB')

        # assume image is alerady in JPEG format
        image = self.hf_dataset[idx]['image'].convert('RGB')
        image = self.transform(image)

        # Convert image to tensor
        image = torch.tensor(np.array(image))

        label = self.hf_dataset[idx]['label']
        return image, label

loss_fn = torch.nn.CrossEntropyLoss()
acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())
torch.autograd.set_detect_anomaly(True)
training_times = []
best_val_acc = 0
def train_mae_mix(train_dataloader_source, train_dataloader_target,
                  mae_mix, mae_optimizer, classifier_optimizer, lr_scheduler_mae, lr_scheduler_classifier,
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

    # if args.mode == 'mae': pretrain on source dataset
    # if args.mode == 'classifier': finetune on target dataset
    # if args.mode == 'mix': mix training on both source and target dataset

    if args.mode == 'mae':
        for images, labels in tqdm(iter(train_dataloader_source)):
            # Prepare image
            images = images.to(args.device)
            labels = labels.to(args.device)
            # Forward pass encoder with the full image
            features = mae_mix.unmasked_generate(images)

            # MAE pretraining step
            # truncate CLS token
            features_truncated = features[:-1, :, :]  # [seq_len, batch_size, embed_dim]

            masked_features, forward_indexes, backward_indexes = mae_mix.apply_mask_to_features(features_truncated)
            reconstructed_img, mask = mae_mix.generate(masked_features, backward_indexes)
            mae_loss = torch.mean((reconstructed_img - images) ** 2 * mask) / args.mask_ratio
            mae_optimizer.zero_grad()
            mae_loss.backward(retain_graph=True)
            # Disable gradient accumulation
            # if step_count % steps_per_update == 0:
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
            # Forward pass encoder with the full image
            features = mae_mix.unmasked_generate(images)
            logits = mae_mix.predict_from_features(features)

            classifier_loss = loss_fn(logits, labels)
            classifier_loss.backward()

            acc = acc_fn(logits, labels)
            # Disable gradient accumulation
            ## if step_count % steps_per_update == 0:

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
                features = mae_mix.unmasked_generate(images_target)
                features_truncated = features[:-1, :, :]  # [seq_len, batch_size, embed_dim]
                masked_features, forward_indexes, backward_indexes = mae_mix.apply_mask_to_features(
                    features_truncated)
                reconstructed_img, mask = mae_mix.generate(masked_features, backward_indexes)
                mae_loss = torch.mean((reconstructed_img - images_target) ** 2 * mask) / args.mask_ratio
                mae_loss.backward(retain_graph=True)
                mae_losses.append(mae_loss.item())
                # then train on target dataset with classification loss
                labels = labels_target
                logits = mae_mix.predict_from_features(features)
                classifier_loss = loss_fn(logits, labels)
                classifier_loss.backward(retain_graph=True)
                acc = acc_fn(logits, labels)
                losses.append(classifier_loss.item())
                acces.append(acc.item())

                mae_optimizer.step()
                classifier_optimizer.step()
                mae_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
            else:
                # Mix images by averaging them
                images = (1-args.lmda) * images_source + args.lmda * images_target
                # Use the target label for the mixed image
                labels = labels_target

                features = mae_mix.unmasked_generate(images)
                labels = labels
                features_truncated = features[:-1, :, :]  # [seq_len, batch_size, embed_dim]

                masked_features, forward_indexes, backward_indexes = mae_mix.apply_mask_to_features(features_truncated)
                reconstructed_img, mask = mae_mix.generate(masked_features, backward_indexes)

                mae_loss = torch.mean((reconstructed_img - images) ** 2 * mask) / args.mask_ratio
                # mae_loss.backward(retain_graph=True)
                mae_losses.append(mae_loss.item())

                logits = mae_mix.predict_from_features(features)
                classifier_loss = loss_fn(logits, labels)
                # classifier_loss.backward()

                loss = args.mix_loss_alpha * mae_loss + (1 - args.mix_loss_alpha) * classifier_loss
                loss.backward()

                acc = acc_fn(logits, labels)
                losses.append(classifier_loss.item())
                acces.append(acc.item())

                mae_optimizer.step()
                classifier_optimizer.step()
                mae_optimizer.zero_grad()
                classifier_optimizer.zero_grad()

    # reset dataloader
    for data in train_dataloader_target:
        pass
    for data in train_dataloader_source:
        pass

    if args.mode not in ['mae', 'classifier', 'kmix']:
        raise Exception('Unknown mode')

    # for images, labels in tqdm(iter(train_dataloader)):
    #     # Prepare image
    #     images = images.to(args.device)
    #     labels = labels.to(args.device)
    #
    #     if args.mode == 'kmix':
    #         assert args.k != 0
    #         '''
    #             k-mix: conduct k times of unsuperivsed pretraining, and then conduct one time of supervised finetuning
    #         '''
    #         # Don't clear gradients for k-mix in the beginning of each step
    #
    #         # Forward pass encoder with the full image
    #         # features = mae_mix.unmasked_generate(images)
    #         # features_list.append((features, labels))
    #         features_list.append((mae_mix.unmasked_generate(images), labels))
    #
    #         # MAE pretraining step
    #         # truncate CLS token
    #         # features_truncated = features[:-1, :, :]  # [seq_len, batch_size, embed_dim]
    #         features_truncated = features_list[-1][0][:-1, :, :]
    #
    #         masked_features, forward_indexes, backward_indexes = mae_mix.apply_mask_to_features(features_truncated)
    #         reconstructed_img, mask = mae_mix.generate(masked_features, backward_indexes)
    #         mae_loss = torch.mean((reconstructed_img - images) ** 2 * mask) / args.mask_ratio
    #         mae_loss.backward(retain_graph=True)
    #         mae_losses.append(mae_loss.item())
    #         # bypass loggging to avoid error
    #         losses.append(0.0)
    #         acces.append(0.0)
    #
    #         if args.debug == True:
    #             print(f'current k is {args.k}')
    #             print(f'current k_index is {k_index}')
    #             print(f'current mae loss is {mae_loss.item()}')
    #
    #         k_index += 1
    #         if k_index == args.k:
    #             k_index = 0
    #             # Classifier finetuning step
    #             # Reuse the features from the full image forward pass
    #             for each in features_list:
    #                 # logits = mae_mix.predict_from_features(features)
    #                 logits = mae_mix.predict_from_features(each[0])
    #                 classifier_loss = loss_fn(logits, labels)
    #                 classifier_loss.backward(retain_graph=True)
    #                 acc = acc_fn(logits, labels)
    #             # TODO: make sure only step pnce for each k-mix
    #             # TODO: memory profiling for k-mix
    #             mae_optimizer.step()
    #             classifier_optimizer.step()
    #
    #             # clean up gradients for next k-mix
    #             mae_optimizer.zero_grad()
    #             classifier_optimizer.zero_grad()
    #             features_list.clear()
    #             losses.append(classifier_loss.item())
    #             acces.append(acc.item())
    #
    #             if args.debug == True:
    #                 print(f'current k is {k_index}')
    #                 print(f'current classifier loss is {classifier_loss.item()}')
    #                 print(f'current mae loss is {mae_loss.item()}')
    #     elif args.mode == 'mae':
    #         # Forward pass encoder with the full image
    #         features = mae_mix.unmasked_generate(images)
    #
    #         # MAE pretraining step
    #         # truncate CLS token
    #         features_truncated = features[:-1, :, :]  # [seq_len, batch_size, embed_dim]
    #
    #         masked_features, forward_indexes, backward_indexes = mae_mix.apply_mask_to_features(features_truncated)
    #         reconstructed_img, mask = mae_mix.generate(masked_features, backward_indexes)
    #         mae_loss = torch.mean((reconstructed_img - images) ** 2 * mask) / args.mask_ratio
    #         mae_optimizer.zero_grad()
    #         mae_loss.backward(retain_graph=True)
    #         # Disable gradient accumulation
    #         # if step_count % steps_per_update == 0:
    #         mae_optimizer.step()
    #         mae_optimizer.zero_grad()
    #         mae_losses.append(mae_loss.item())
    #
    #         losses.append(0.0)
    #         acces.append(0.0)
    #
    #         if args.debug == True:
    #             print(f'current mae loss is {mae_loss.item()}')
    #     elif args.mode == 'classifier':
    #         # Forward pass encoder with the full image
    #         features = mae_mix.unmasked_generate(images)
    #         logits = mae_mix.predict_from_features(features)
    #
    #         classifier_loss = loss_fn(logits, labels)
    #         classifier_loss.backward()
    #
    #         acc = acc_fn(logits, labels)
    #         # Disable gradient accumulation
    #         ## if step_count % steps_per_update == 0:
    #
    #         classifier_optimizer.step()
    #         classifier_optimizer.zero_grad()
    #         losses.append(classifier_loss.item())
    #         acces.append(acc.item())
    #
    #         mae_losses.append(0.0)
    #
    #         if args.debug == True:
    #             print(f'current classifier loss is {classifier_loss.item()}')
    #     else:
    #         raise Exception('Unknown mode')


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
            logits = mae_mix.predict(images)
            # Compute the accuracy
            loss = loss_fn(logits, labels)
            acc = acc_fn(logits, labels)
            losses.append(loss.item())
            acces.append(acc.item())
            # Log the accuracy to wandb
            # Log the accuracy to tensorboard
            # Save the model if the accuracy is the best so far
            torch.save(mae_mix, args.output_model_path)
        avg_val_loss = sum(losses) / len(losses)
        avg_val_acc = sum(acces) / len(acces)


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
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
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
    parser.add_argument('--model_name', type=str, default='vit-t', choices=['vit-t'])
    parser.add_argument('--lmda', type=float, default=0.5) # lambda is reserved keyword for python, use lmda instead
    parser.add_argument('--data_mix_noise', action='store_true')
    parser.add_argument('--data_no_mix', action='store_true')

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
                transforms.Resize((224, 224), antialias=True,
                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_validate = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True,
                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
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
                transforms.Resize((224, 224), antialias=True,
                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
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
                transforms.Resize((224, 224), antialias=True,
                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
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
                transforms.Resize((224, 224), antialias=True,
                                  interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
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
        transforms.Resize((64, 64), antialias=True),
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

    train_dataloader_source = torch.utils.data.DataLoader(train_dataset_source, args.load_batch_size, shuffle=True,
                                                          num_workers=32)
    train_dataloader_target = torch.utils.data.DataLoader(train_dataset_target, args.load_batch_size, shuffle=True,
                                                          num_workers=32)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.load_batch_size, shuffle=False, num_workers=32)

    # Model initialization
    # For standard AE pretraining, we set mask_ratio to 0.0.
    # if args.pretrained_model_path is None:
    #     if args.dataset == 'cifar10':
    #         mae_mix = MAE_Mix(mae_encoder=None, emb_dim=192, image_size=32, patch_size=2, num_classes=10, decoder_layer=2, mask_ratio=0.0).to(device)
    #     elif args.dataset == 'cifar100':
    #         mae_mix = MAE_Mix(mae_encoder=None, emb_dim=192, image_size=32, patch_size=2, num_classes=100, decoder_layer=2, mask_ratio=0.0).to(device)
    #     elif args.dataset == 'svhn':
    #         mae_mix = MAE_Mix(mae_encoder=None, emb_dim=192, image_size=32, patch_size=2, num_classes=10, decoder_layer=2, mask_ratio=0.0).to(device)
    #     elif args.dataset == 'tinyimagenet':
    #         mae_mix = MAE_Mix(mae_encoder=None, emb_dim=192, image_size=64, patch_size=4, num_classes=200, decoder_layer=2, mask_ratio=0.0).to(device)
    #     elif args.dataset == 'imagenet':
    #         mae_mix = MAE_Mix(mae_encoder=None, emb_dim=192, image_size=224, patch_size=16, num_classes=1000, decoder_layer=2, mask_ratio=0.0).to(device)
    #     else:
    #         raise Exception('Unknown dataset')

    # hard code input resolution to 64x64, and output class to target dataset class number for now
    if args.dataset == 'cifar10':
        mae_mix = MAE_Mix(mae_encoder=None, emb_dim=192, image_size=64, patch_size=4, num_classes=10, decoder_layer=2, mask_ratio=0.0).to(device)
    elif args.dataset == 'cifar100':
        mae_mix = MAE_Mix(mae_encoder=None, emb_dim=192, image_size=64, patch_size=4, num_classes=100, decoder_layer=2, mask_ratio=0.0).to(device)

    # profile params and flops
    if args.profile:
        input = torch.randn(1, 3, 224, 224).to(device)
        macs, params = profile(mae_mix, inputs=(input, ))
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
    mae_parameter_groups = list(mae_mix.encoder.parameters()) + list(mae_mix.decoder.parameters())
    classifier_parameter_groups = list(mae_mix.encoder.parameters()) + list(mae_mix.classifier.parameters())

    if args.dataset == 'tinyimagenet':
        # Pre-Training Optimizer Configuration
        mae_optimizer = torch.optim.AdamW(mae_parameter_groups,
                                          lr=5e-4 * args.load_batch_size / 128,  # base learning rate for pre-training
                                          betas=(0.9, 0.95),  # beta values for pre-training
                                          weight_decay=0.15)  # weight decay for pre-training

        # Fine-Tuning Optimizer Configuration
        classifier_optimizer = torch.optim.AdamW(classifier_parameter_groups,
                                                 lr=1e-3 * args.load_batch_size / 128,
                                                 # base learning rate for fine-tuning
                                                 betas=(0.9, 0.999),  # beta values for fine-tuning
                                                 weight_decay=0.05)  # weight decay for fine-tuning
    else:
        mae_optimizer = torch.optim.AdamW(mae_parameter_groups, lr=1.5e-4 * args.load_batch_size / 256 * args.k, betas=(0.9, 0.95),
                                          weight_decay=args.weight_decay)
        classifier_optimizer = torch.optim.AdamW(classifier_parameter_groups, lr=1e-3 * args.load_batch_size / 256 * args.k,
                                                 betas=(0.9, 0.95),
                                                 weight_decay=args.weight_decay)

    if args.pt_epoch == 0:
        # enforce no pretraining
        lr_func_mae = lambda epoch: 1
    else:
        lr_func_mae = lambda epoch: min((epoch + 1) / (args.pt_warmup_epoch + 1e-8),
                                    0.5 * (math.cos(epoch / args.pt_epoch * math.pi) + 1))

    lr_func_mix = lambda epoch: min((epoch + 1) / (args.ft_warmup_epoch + 1e-8),
                                    0.5 * (math.cos(epoch / args.ft_epoch * math.pi) + 1))

    lr_scheduler_mae = torch.optim.lr_scheduler.LambdaLR(mae_optimizer, lr_lambda=lr_func_mae, verbose=True)
    lr_scheduler_classifier = torch.optim.lr_scheduler.LambdaLR(classifier_optimizer, lr_lambda=lr_func_mix,
                                                                verbose=True)

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
                                                        mae_mix, mae_optimizer, classifier_optimizer, lr_scheduler_mae, lr_scheduler_classifier, args)
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