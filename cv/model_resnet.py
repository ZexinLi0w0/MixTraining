import torch
import timm
import torch.nn as nn

from abstract_model import AbstractModel

class Decoder(AbstractModel):
    def __init__(self, num_features, resolution):
        super(Decoder, self).__init__()

        # Calculate the appropriate kernel size and stride for the first layer
        # These values depend on the final resolution
        kernel_size_1 = resolution // 2
        stride_1 = resolution // 2

        self.conv_transpose1 = nn.ConvTranspose2d(
            in_channels=num_features,
            out_channels=64,
            kernel_size=kernel_size_1,
            stride=stride_1,
            padding=0
        ) # Output size: [batch_size, 64, resolution/2, resolution/2]

        self.conv_transpose2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1
        ) # Output size: [batch_size, 3, resolution, resolution]

        self.activation = nn.ReLU()

    def forward(self, x, mode="train"):
        # bypass mode parameter here
        x = self.conv_transpose1(x)
        x = self.activation(x)
        x = self.conv_transpose2(x)
        return x

    def initialize_weights(self):
        # use kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class MixModel(AbstractModel):
    def __init__(self, backbone_name='resnet18', num_classes=10, pretrained=True, resolution=32):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        self.num_features = self.backbone.num_features
        # remove the gloabl pooling layer and the classifier layer
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-2])

        self.head = torch.nn.Sequential(
            # (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
            # (fc): Linear(in_features=512, out_features=1000, bias=True)
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Flatten(),
            torch.nn.Linear(self.num_features, num_classes),
        )

        # mix with a decoder to reconstruct the image, use several convtranspose layers
        self.decoder = Decoder(num_features=self.num_features, resolution=resolution)

        # if not pretrained model, using kaiming initialization
        if not pretrained:
            self.initialize_weights()


    def forward(self, img, mode="predict"):
        assert mode in ["get_feature", "predict", "generate", "predict_from_features", "generate_from_features"]
        if mode == "get_feature":
            return self.get_features(img)
        elif mode == "predict":
            return self.predict(img)
        elif mode == "generate":
            return self.generate(img)
        elif mode == "predict_from_features":
            return self.predict_from_features(img)
        elif mode == "generate_from_features":
            return self.generate_from_features(img)
        else:
            raise NotImplementedError

    def initialize_weights(self):
        # Initialize weights for the backbone and head if not pretrained
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone_and_decoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def predict(self, img):
        features = self.backbone(img)
        logits = self.head(features)
        return logits

    def generate(self, img):
        features = self.backbone(img)
        reconstructed_img = self.decoder(features)
        return reconstructed_img

    def get_features(self, img):
        features = self.backbone(img)
        return features

    def generate_from_features(self, features):
        reconstructed_img = self.decoder(features)
        return reconstructed_img

    def predict_from_features(self, features):
        logits = self.head(features)
        return logits

if __name__ == '__main__':
    model = MixModel('resnet18', 10, False)
    model.eval()
    img = torch.randn(1, 3, 32, 32)
    logits = model(img, mode="predict")
    print(logits.shape)

    reconstructed_img = model(img, mode="generate")
    print(reconstructed_img.shape)

    features = model(img, mode="get_feature")
    print(features.shape)
    reconstructed_img = model(features, mode="generate_from_features")
    print(reconstructed_img.shape)
    logits = model(features, mode="predict_from_features")
    print(logits.shape)

    import timm

    # Load the ResNet-18 model from timm
    timm_resnet18 = timm.create_model('resnet18', pretrained=False)
    # Counting parameters
    timm_params = sum(p.numel() for p in timm_resnet18.parameters() if p.requires_grad)
    print("Timm ResNet-18 parameters: {}".format(timm_params))

    import torchvision

    # Load the ResNet-18 model from torchvision
    torchvision_resnet18 = torchvision.models.resnet18(weights=None)

    # Counting parameters
    torchvision_params = sum(p.numel() for p in torchvision_resnet18.parameters() if p.requires_grad)
    print("Torchvision ResNet-18 parameters: {}".format(torchvision_params))