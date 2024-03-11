import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

from abstract_model import AbstractModel

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(AbstractModel):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def initialize_weights(self):
        pass

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(AbstractModel):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.mask_ratio = mask_ratio

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.initialize_weights()

    def initialize_weights(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img, mode='default'):
        if mode == 'default':
            self.shuffle.ratio = self.mask_ratio
            patches = self.patchify(img)
            patches = rearrange(patches, 'b c h w -> (h w) b c')
            patches = patches + self.pos_embedding

            patches, forward_indexes, backward_indexes = self.shuffle(patches)

            patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
            patches = rearrange(patches, 't b c -> b t c')
            features = self.layer_norm(self.transformer(patches))
            features = rearrange(features, 'b t c -> t b c')

            return features, backward_indexes
        elif mode == 'mask_forward':
            return self.mask_forward(img)
        elif mode == 'unmask_forward':
            return self.unmask_forward(img)
        else:
            raise NotImplementedError

    def mask_forward(self, img):
        # return all info
        self.shuffle.ratio = self.mask_ratio
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, forward_indexes, backward_indexes

    def unmask_forward(self, img):
        # return all info
        self.shuffle.ratio = 0.0
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        # Remove shuffle
        # patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features

class MAE_Decoder(AbstractModel):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.initialize_weights()

    def initialize_weights(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    # Keep it unchanged
    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(AbstractModel):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def initialize_weights(self):
        pass

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask

class ViT_Classifier(AbstractModel):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def initialize_weights(self):
        pass

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits

class MAE_Mix(AbstractModel):
    def __init__(self, mae_encoder=None, num_classes=10,
                 image_size=32, patch_size=2, emb_dim=192,
                 encoder_layer=12, encoder_head=3,
                 decoder_layer=4, decoder_head=3,
                 mask_ratio=0.75):
        super().__init__()
        if mae_encoder is None:
            self.encoder = MAE_Encoder(image_size, patch_size, emb_dim,
                                       encoder_layer, encoder_head, mask_ratio)
        else:
            self.encoder = mae_encoder
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim,
                                   decoder_layer, decoder_head)
        self.classifier = torch.nn.Linear(emb_dim, num_classes)

    def initialize_weights(self):
        pass

    def forward(self, img, mode='predict'):
        # Provides classification predictions from the image features
        if mode == 'predict':
            return self.predict(img)
        elif mode == 'generate':
            return self.unmasked_generate(img)
        elif mode == 'mask_generate':
            return self.masked_generate(img)
        elif mode == 'generate_from_features':
            # TODO: tofix
            return self.generate(img)
        elif mode == 'predict_from_features':
            return self.predict_from_features(img)
        elif mode == 'full_generate':
            return self.full_generate(img)

    def apply_mask_to_features(self, features):
        # Mask the latent features
        self.encoder.shuffle.ratio = self.encoder.mask_ratio
        masked_features, forward_indexes, backward_indexes = self.encoder.shuffle(features)
        masked_features = torch.cat([self.encoder.cls_token.expand(-1, masked_features.shape[1], -1), masked_features], dim=0)
        return masked_features, forward_indexes, backward_indexes

    def unmasked_generate(self, img):
        # Generates masked image features
        features = self.encoder.unmask_forward(img)
        return features

    def masked_generate(self, img):
        # Generates masked image features
        features, forward_indexes, backward_indexes = self.encoder.mask_forward(img)
        return features, backward_indexes

    def generate(self, features, backward_indexes):
        # Reconstructs the image from the masked image features
        reconstructed_img, mask = self.decoder(features, backward_indexes)
        return reconstructed_img, mask

    def predict(self, img):
        # Provides classification predictions from the image features
        features = self.encoder.unmask_forward(img)
        logits = self.classifier(features[0])
        return logits

    def predict_from_features(self, features):
        # If 'features' is a batch of sequences, and we want the CLS token, it is usually at index 0
        cls_tokens = features[0]  # Get the CLS token for each item in the batch
        logits = self.classifier(cls_tokens)
        return logits

    def full_generate(self, img):
        # Generates a full image from the masked image features
        features, backward_indexes = self.masked_generate(img)
        reconstructed_img, mask = self.generate(features, backward_indexes)
        return reconstructed_img, mask

if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)

    mae = MAE_Mix()
    import pdb; pdb.set_trace()