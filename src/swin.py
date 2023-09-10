from transformers import SwinBackbone, SwinConfig
from image import train_loader
from einops import rearrange
import torchvision
import torch
import torch.nn as nn


class PatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = rearrange(x, "B H W (P1 P2 C) -> B (H P1) (W P2) C", P1=2, P2=2)
        x = self.norm(x)
        return x


class SwinEncoder(nn.Module):
    def __init__(self, **kwargs):
        """
        kwargs:
            image_size (int, optional, defaults to 224) — The size (resolution) of each image.
            patch_size (int, optional, defaults to 4) — The size (resolution) of each patch.
            num_channels (int, optional, defaults to 3) — The number of input channels.
            embed_dim (int, optional, defaults to 96) — Dimensionality of patch embedding.
            depths (list(int), optional, defaults to [2, 2, 6, 2]) — Depth of each layer in the Transformer encoder.
            num_heads (list(int), optional, defaults to [3, 6, 12, 24]) — Number of attention heads in each layer of the Transformer encoder.
            window_size (int, optional, defaults to 7) — Size of windows.
            mlp_ratio (float, optional, defaults to 4.0) — Ratio of MLP hidden dimensionality to embedding dimensionality.
            qkv_bias (bool, optional, defaults to True) — Whether or not a learnable bias should be added to the queries, keys and values.
            hidden_dropout_prob (float, optional, defaults to 0.0) — The dropout probability for all fully connected layers in the embeddings and encoder.
            attention_probs_dropout_prob (float, optional, defaults to 0.0) — The dropout ratio for the attention probabilities.
            drop_path_rate (float, optional, defaults to 0.1) — Stochastic depth rate.
            hidden_act (str or function, optional, defaults to "gelu") — The non-linear activation function (function or string) in the encoder. If string, "gelu", "relu", "selu" and "gelu_new" are supported.
            use_absolute_embeddings (bool, optional, defaults to False) — Whether or not to add absolute position embeddings to the patch embeddings.
            initializer_range (float, optional, defaults to 0.02) — The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (float, optional, defaults to 1e-12) — The epsilon used by the layer normalization layers.
            encoder_stride (int, optional, defaults to 32) — Factor to increase the spatial resolution by in the decoder head for masked image modeling.
            out_features (List[str], optional) — If used as backbone, list of features to output. Can be any of "stem", "stage1", "stage2", etc. (depending on how many stages the model has). If unset and out_indices is set, will default to the corresponding stages. If unset and out_indices is unset, will default to the last stage.
            out_indices (List[int], optional) — If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how many stages the model has). If unset and out_features is set, will default to the corresponding stages. If unset and out_features is unset, will default to the last stage.
        """
        super().__init__()

        if len(kwargs) == 0:
            self.config = SwinConfig()
        else:
            self.config = SwinConfig(kwargs)

        self.model = SwinBackbone(self.config)

    def forward(self, x):
        return self.model(x)


class Expand(nn.Module):
    def __init__(self, target_dim: int = 3):
        super().__init__()
        self.dim = target_dim

    def forward(self, x):
        try:
            inp = rearrange(x.feature_maps[0], "B C H W -> B H W C")
        except AttributeError:
            inp = rearrange(x, "B C H W -> B H W C")
        _dim = inp.size()[-1]
        while _dim > self.dim:
            expand = PatchExpanding(_dim)
            inp = expand(inp)
            _dim = inp.size()[-1]

        return rearrange(inp, "B H W C -> B C H W")


class SwinDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand_1 = Expand(12)

        # We can build an architecture here
        self.expand_2 = Expand(3)
        return

    def forward(self, x):
        out = self.expand_1(x)
        out = self.expand_2(out)
        return out


model = SwinBackbone(SwinConfig())
transform = torchvision.transforms.ToPILImage()
dataloader = train_loader(resize=224)

if __name__ == "__main__":
    encoder = SwinEncoder()
    decoder = SwinDecoder()
    batch = next(iter(dataloader))
    out = encoder(batch)
    out = decoder(out)

    breakpoint()
