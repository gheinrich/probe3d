from enum import Enum
import math
import os
import sys
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from timm.models import Eva
from timm.layers import to_2tuple

#from input_conditioner import InputConditioner
#from visualize import get_pca_map, draw_grid


class EvaAdapter(nn.Module):
    def __init__(self, model: Eva):
        super().__init__()
        self.inner = model

    def forward(self, *args, **kwargs):
        features = self.inner.forward_features(*args, **kwargs)
        head = self.inner.forward_head(features)

        if self.inner.global_pool:
            if self.inner.global_pool != 'avg':
                features = features[:, 1:]

        return head, features


def get_teacher_adapter(model: nn.Module) -> nn.Module:
    if isinstance(model, Eva):
        return EvaAdapter(model)
    return model


class AdaptorRegistry:
    methods = dict()

    @staticmethod
    def register(keys: Union[Any, List[Any]], ctor):
        if not isinstance(keys, (tuple, list)):
            keys = [keys]
        for key in keys:
            AdaptorRegistry.methods[key] = ctor

    @staticmethod
    def get_adaptor(key: Optional[str] = None):
        return AdaptorRegistry.methods[key.lower() if key is not None else None]


def register_adaptor(*keys):
    def decorator(cls):
        AdaptorRegistry.register(keys, cls)
        return cls
    return decorator


@register_adaptor(None, 'none')
def _identity(model, *args, **kwargs):
    return model


#@register_adaptor('mosaic')
#class MosaicAdaptor(nn.Module):
#    def __init__(self, model: nn.Module, outer_input_size: int, inner_input_size: int, downsample_size: int,
#                 conditioner: InputConditioner, step_size: int = 0, debug: bool = False, jitter: bool = False,
#                 **kwargs):
#        super().__init__()
#        self.inner = model
#        self.input_size = to_2tuple(outer_input_size)
#        self._inner_input_size = to_2tuple(inner_input_size)
#        self.downsample_size = downsample_size
#        self.step_size = step_size
#        self._conditioner = conditioner
#        self._debug = debug
#        self._counter = 0
#        self.jitter = jitter
#
#    @torch.no_grad()
#    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#        h, w = images.shape[-2:]
#
#        num_per_row = self._inner_input_size[1] // w
#        num_per_col = self._inner_input_size[0] // h
#        num_per_img = num_per_row * num_per_col
#        batch_size = int(math.ceil(images.shape[0] / num_per_img))
#
#        tile_x = self._inner_input_size[1] // num_per_row
#        tile_y = self._inner_input_size[0] // num_per_col
#
#        num_x_cells_per_tile = tile_x // self.downsample_size
#        num_y_cells_per_tile = tile_y // self.downsample_size
#
#        num_h_cells = h // self.downsample_size
#        num_w_cells = w // self.downsample_size
#
#        hr_buff = torch.zeros(batch_size, 3, *self._inner_input_size, dtype=images.dtype, device=images.device)
#        offsets = []
#        for i, img in enumerate(images):
#            b_idx = i // num_per_img
#            i -= b_idx * num_per_img
#
#            r = i // num_per_row
#            c = i % num_per_row
#
#            if self.jitter:
#                offset_x = np.random.randint(0, num_x_cells_per_tile - num_w_cells + 1)
#                offset_y = np.random.randint(0, num_y_cells_per_tile - num_h_cells + 1)
#            else:
#                offset_x = offset_y = 0
#
#            offsets.append((offset_x, offset_y))
#
#            x = c * tile_x + offset_x * self.downsample_size
#            y = r * tile_y + offset_y * self.downsample_size
#
#            hr_buff[b_idx, :, y:y+h, x:x+w] = img
#
#        all_summary, all_features = self.inner(hr_buff)
#
#        all_features = rearrange(all_features, 'b (h w) c -> b h w c',
#                                 h=self._inner_input_size[0] // self.downsample_size,
#                                 w=self._inner_input_size[1] // self.downsample_size).float()
#
#        if self._debug:
#            dn_buff = _denormalize_images(hr_buff, self._conditioner).permute(0, 2, 3, 1)
#            pca_viz = torch.cat([
#                get_pca_map(feats, self._inner_input_size)
#                for feats in all_features
#            ], dim=0)
#
#            os.makedirs('mosaic', exist_ok=True)
#            dbg_imgs = torch.cat([dn_buff, pca_viz], dim=-2).mul_(255).byte().cpu()
#            for img in dbg_imgs:
#                draw_grid(img, spacing=16, color=0, layout='hwc')
#                cv2.imwrite(f'mosaic/viz_{self._counter}.jpg', cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR))
#                self._counter += 1
#
#        # Just give every sub-image the same summary
#        summary = torch.repeat_interleave(all_summary, num_per_img, dim=0).float()[:images.shape[0]]
#
#        tile_x //= self.downsample_size
#        tile_y //= self.downsample_size
#
#        op_w = w // self.downsample_size
#        op_h = h // self.downsample_size
#
#        features = []
#        for i in range(images.shape[0]):
#            b_idx = i // num_per_img
#            i -= b_idx * num_per_img
#
#            r = i // num_per_row
#            c = i % num_per_row
#
#            offset_x, offset_y = offsets[i]
#
#            x = c * tile_x + offset_x
#            y = r * tile_y + offset_y
#
#
#            window = all_features[b_idx, y:y+op_h, x:x+op_w]
#            features.append(window)
#
#        features = torch.stack(features, dim=0).flatten(1, 2)
#
#        return summary, features
#
#
#def _denormalize_images(images: torch.Tensor, conditioner: InputConditioner):
#    return images * conditioner.norm_std + conditioner.norm_mean
#
#
#@register_adaptor('crop_window')
#class CropWindowAdaptor(nn.Module):
#    def __init__(self, model: nn.Module, outer_input_size: int, inner_input_size: int, downsample_size: int,
#                 conditioner: InputConditioner, step_size: int = 0, debug: bool = False, jitter: bool = False,
#                 **kwargs):
#        super().__init__()
#        self.inner = model
#        self.input_size = to_2tuple(outer_input_size)
#        self._inner_input_size = to_2tuple(inner_input_size)
#        self.downsample_size = downsample_size
#        self.step_size = step_size
#        self._conditioner = conditioner
#        self._debug = debug
#        self._counter = 0
#        self.jitter = jitter
#
#    @torch.no_grad()
#    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#        b, _, h, w = images.shape
#
#        num_x_cells = self.input_size[1] // self.downsample_size
#        num_y_cells = self.input_size[0] // self.downsample_size
#
#        hr_buff = torch.zeros(b, 3, *self._inner_input_size, dtype=images.dtype, device=images.device)
#        hr_buff[:, :, :h, :w] = images
#
#        all_summary, all_features = self.inner(hr_buff)
#
#        all_features = rearrange(all_features, 'b (h w) c -> b h w c',
#                                 h=self._inner_input_size[0] // self.downsample_size,
#                                 w=self._inner_input_size[1] // self.downsample_size).float()
#
#        all_features = all_features[:, :num_y_cells, :num_x_cells]
#        all_features = all_features.flatten(1, 2)
#
#        return all_summary, all_features


class UpsampleMethodBase(nn.Module):
    def __init__(self, model: nn.Module, outer_input_size: int, inner_input_size: int, downsample_size: int,
                 feature_dim: int, **kwargs):
        super().__init__()
        self.model = model
        self.input_size = to_2tuple(outer_input_size)
        self.inner_input_size = to_2tuple(inner_input_size)
        self.downsample_size = downsample_size
        self.student_patch_size = downsample_size
        self._feature_dim = feature_dim

    @property
    def num_down_patches(self):
        return tuple(s // self.downsample_size for s in self.inner_input_size)

    @property
    def num_up_patches(self):
        return tuple(s // self.student_patch_size for s in self.input_size)

    @property
    def patch_size(self):
        return self.downsample_size

    @property
    def feature_dim(self):
        return self._feature_dim

    def to_lower(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.inner_input_size, mode='bilinear', align_corners=False)
        y = self.model(x)
        return y

    def to_higher(self, x: torch.Tensor) -> torch.Tensor:
        summary, features = self.to_lower(x)
        features = rearrange(features, 'b (h w) c -> b c h w', h=self.num_down_patches[0], w=self.num_down_patches[1])
        features = F.interpolate(features, size=self.num_up_patches, mode='bilinear', align_corners=False)
        # features = rearrange(features, 'b c h w -> b (h w) c')
        return summary, features


@register_adaptor('to_lower')
class UpsampleToLower(UpsampleMethodBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.to_lower(x)
        return y


@register_adaptor('to_higher')
class UpsampleToHigher(UpsampleMethodBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.to_higher(x)
        return y


@register_adaptor('tile')
class UpsampleTile(UpsampleMethodBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tiles = tuple(int(math.ceil(h / l)) for h, l in zip(self.input_size, self.inner_input_size))
        self.rs_input_size = tuple(d * t for d, t in zip(self.inner_input_size, self.num_tiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tiles = self.tiled_inference(x)
        return tiles

    def tiled_inference(self, x: torch.Tensor) -> torch.Tensor:
        x_low = F.interpolate(x, size=self.inner_input_size, mode='bilinear', align_corners=False)

        x = F.interpolate(x, size=self.rs_input_size, mode='bilinear', align_corners=False)

        x_tiled = rearrange(x, 'b c (th h) (tw w) -> (b th tw) c h w',
                            th=self.num_tiles[0], tw=self.num_tiles[1], h=self.inner_input_size[0], w=self.inner_input_size[1])

        b = x.shape[0]
        x_tiled = torch.cat([x_low, x_tiled], dim=0)

        y_summary, y_tiled_features = self.model(x_tiled)

        y_summary = y_summary[:b]
        y_tiled_features = y_tiled_features[b:]

        y_tiled_features = rearrange(y_tiled_features, '(b th tw) (h w) c -> b c (th h) (tw w)',
                                     b=x.shape[0], th=self.num_tiles[0], tw=self.num_tiles[1],
                                     h=self.num_down_patches[0], w=self.num_down_patches[1])

        y_tiled_features = F.interpolate(y_tiled_features, size=self.num_up_patches, mode='bilinear', align_corners=False)
        #y_tiled_features = rearrange(y_tiled_features, 'b c h w -> b (h w) c')
        return y_summary, y_tiled_features


@register_adaptor('s2')
class UpsampleS2(UpsampleTile):
    def __init__(self, *args, embed_dim: int, feature_dim: int, beta: float = 0.8, concatenate: bool = False, **kwargs):
        if concatenate:
            feature_dim *= 2
        super().__init__(*args, feature_dim=feature_dim, **kwargs)
        self.beta = beta
        self.concatenate = concatenate
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_summary, h_features = self.to_higher(x)
        t_summary, t_features = self.tiled_inference(x)

        summary = self.beta * h_summary + (1 - self.beta) * t_summary
        if self.concatenate:
            features = torch.cat([h_features, t_features], dim=-1)
        else:
            features = (1 - self.beta) * h_features + self.beta * t_features
        return summary, features


@register_adaptor('s2_cat')
def upsample_s2_cat(*args, **kwargs):
    return UpsampleS2(*args, **kwargs, concatenate=True)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs/FeatUp'))

class UpsWrapper(nn.Module):
    def __init__(self, method: nn.Module):
        super().__init__()
        self.input_size = method.inner_input_size[0]
        self.patch_size = method.patch_size
        self.embed_dim = method.feature_dim
        self.model = method.model

    def forward(self, x: torch.Tensor, return_summary: bool = False):
        assert return_summary
        summary, features = self.model(x)

        h, w = tuple(d // self.patch_size for d in x.shape[-2:])
        features = rearrange(features, 'b (h w) c -> b c h w', h=h, w=w)

        return summary, features

@register_adaptor('featsharp')
class UpsampleFeatSharp(nn.Module):
    def __init__(self, model: nn.Module, outer_input_size: int, inner_input_size: int, downsample_size: int,
                 feature_dim: int, upsampler_checkpoint: str, do_upsample: bool = True, **kwargs):
        super().__init__()

        self.inner_input_size = to_2tuple(inner_input_size)
        self._input_size_orig = to_2tuple(outer_input_size)
        self.downsample_size = downsample_size
        self.student_patch_size = downsample_size
        self._feature_dim = feature_dim
        self.model = model

        from featup.builder import load_from_file

        wrapper = UpsWrapper(self)
        self.upsampler = load_from_file(upsampler_checkpoint, wrapper, do_upsample=do_upsample)
        pass

    @property
    def patch_size(self):
        return self.downsample_size

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def input_size(self):
        return tuple(d * self.upsampler.input_upsample_factor for d in self.inner_input_size)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lr_y, hr_y, summary = self.upsampler(x, denormalize=True, return_summary=True)

        #hr_y = rearrange(hr_y, 'b c h w -> b (h w) c')

        return summary, hr_y
