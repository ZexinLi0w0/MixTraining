from model_resnet import MixModel

def main():
    import torch
    import torchvision
    from torchvision import transforms
    from torchvision.transforms import ToTensor, Compose, Normalize
    from tqdm import tqdm
    from utils import WarmupCosineLR


    model = MixModel('resnet18', 10, False)

    def _smoke_test(model):

        print("[smoke testing] training on cifar10")
        # load cifar10 datasets
        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        batch_size = 128

        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=64,
                                                       pin_memory=True, prefetch_factor=4)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=64,
                                                     pin_memory=True, prefetch_factor=4)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        epochs = 100

        # Setup scheduler
        warmup_epochs = 20  # Number of warm-up epochs
        min_lr = 1e-6  # Minimum learning rate at the end
        scheduler = WarmupCosineLR(optimizer, warmup_epochs, epochs, min_lr)

        # train the model for 200 epoch
        for e in range(epochs):
            print("==========epoch {}==========".format(e))
            model.train()
            for img, label in iter(train_dataloader):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = torch.nn.CrossEntropyLoss()(logits, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            # test the model
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            criterion = torch.nn.CrossEntropyLoss()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                val_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    _smoke_test(model)

if __name__ == "__main__":
    main()