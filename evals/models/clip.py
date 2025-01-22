from __future__ import annotations

import os

import numpy as np
from pathlib import Path
from typing import List

import einops as E
import open_clip
import torch
from torch import nn
from PIL import Image

from .utils import center_padding, resize_pos_embed, tokens_to_output
from .upsamplers import UpsampleFeatSharp, UpsampleTile, UpsampleToHigher
#from featup.upsamplers import UpsampleFeatSharp, UpsampleTile, UpsampleToHigher
from .visualize_features import get_pca_map, UnNormalize

# Register the model config
open_clip.add_model_config(Path(__file__).parent / "model_configs")


class CLIPWrapper(nn.Module):
    def __init__(self, visual: nn.Module, patch_size: int, multi_layers: List[int], output: str):
        super().__init__()
        self.visual = visual
        self.patch_size = patch_size
        self.multilayers = multi_layers
        self.output = output

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:

        images = center_padding(images, self.patch_size)

        print("input shape", images.shape)

        # clip stuff
        x = self.visual.conv1(images)
        x_hw = x.shape[-2:]
        x = E.rearrange(x, "b c h w -> b (h w) c")

        # concat cls token
        _cls_embed = E.repeat(self.visual.class_embedding, "c -> b 1 c", b=x.shape[0])
        x = torch.cat([_cls_embed.to(x.dtype), x], dim=1)

        # add pos embed
        pos_embed = resize_pos_embed(self.visual.positional_embedding, x_hw)
        x = self.visual.ln_pre(x + pos_embed.to(x.dtype))

        embeds = []
        for i, blk in enumerate(self.visual.transformer.resblocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        summaries = []
        outputs = []
        for i, _x in enumerate(embeds):
            summary = _x[:, 0]
            features = _x[:, 1:]
            summaries.append(summary)
            outputs.append(features)

        return (summaries[-1], outputs[-1]) if len(self.multilayers) == 1 else (summaries, outputs)


class CLIP(nn.Module):
    def __init__(
        self,
        arch="ViT-B-16",
        checkpoint="openai",
        output="dense",
        layer=-1,
        return_multilayer=False,
        featsharp=False,
        feattile=False,
        feattohigher=False,
        featsharp_checkpoint=None,
        featup_outer_input_size=1134,
        featup_inner_input_size=378,
    ):
        super().__init__()
        assert output in ["dense-cls", "cls", "gap", "dense"]
        self.output = output
        self.checkpoint_name = "clip_" + arch.replace("-", "").lower() + checkpoint

        # Initialize a pre-trained CLIP image encoder and freeze it.
        _clip_model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=checkpoint
        )
        _clip_model = _clip_model.eval().to(torch.float32)
        self.visual = _clip_model.visual
        del _clip_model

        # Extract some attributes from CLIP module for easy access.
        self.patch_size = self.visual.conv1.stride[0]

        # get feature dimension
        feat_dim = self.visual.transformer.width
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim
        if return_multilayer:
            feat_dims = [feat_dim, feat_dim, feat_dim, feat_dim]
        else:
            feat_dims = [feat_dim]

        # get extraction targets
        n_layers = len(self.visual.transformer.resblocks)
        multilayers = [
            n_layers // 4 - 1,
            n_layers // 2 - 1,
            n_layers // 4 * 3 - 1,
            n_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = feat_dims
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dims
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        self.visual_wrapper = CLIPWrapper(self.visual, self.patch_size, self.multilayers, self.output)
        if featsharp:
            #import torch._dynamo
            torch._dynamo.config.optimize_ddp=False
            #torch._dynamo.config.suppress_errors = True

            assert featsharp_checkpoint is not None
            self.visual_wrapper = UpsampleFeatSharp(
                model=self.visual_wrapper,
                outer_input_size=featup_outer_input_size,
                inner_input_size=featup_inner_input_size,
                downsample_size=self.patch_size,
                feature_dim=feat_dim,
                upsampler_checkpoint=featsharp_checkpoint,
            )
        elif feattile:
            self.visual_wrapper = UpsampleTile(
                model=self.visual_wrapper,
                outer_input_size=featup_outer_input_size,
                inner_input_size=featup_inner_input_size,
                downsample_size=self.patch_size,
                feature_dim=feat_dim,
            )
        elif feattohigher:
            self.visual_wrapper = UpsampleToHigher(
                model=self.visual_wrapper,
                outer_input_size=featup_outer_input_size,
                inner_input_size=featup_inner_input_size,
                downsample_size=self.patch_size,
                feature_dim=feat_dim,
            )

        self.layer = "-".join(str(_x) for _x in self.multilayers)

        self.sample_count = 0

        self.debug = False

    def forward(self, images):

        print("images feats", images.shape)
        summaries, features = self.visual_wrapper(images)
        print("featshape feats", features.shape)

        if len(self.multilayers) == 1:
            features = [features]
            summaries = [summaries]

        outputs = []
        for summary, patches in zip(summaries, features):
            # If the features are in the shape of (B, N, C), we need to convert to (B, C, H, W)
            if patches.dim() == 3:
                image_size = int(patches.shape[1] ** 0.5)
                assert image_size * image_size == patches.shape[1], f"Invalid number of patches: {patches.shape[1]}"
                out_hw = (image_size, image_size)
                patches = tokens_to_output(self.output, patches, summary, out_hw)
            outputs.append(patches)

        # Save visualization of features on RANK 0.
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if self.debug and self.sample_count % 10 == 0:

                unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                image = images[0]
                print("image", image.shape, image.min(), image.max())
                image = unorm(image)
                print("denorm image", image.shape, image.min(), image.max())
                image = image.permute(1, 2, 0).float() * 255
                image = Image.fromarray(image.cpu().numpy().astype(np.uint8))
                image.save(os.path.join("debug/", f"sample_{self.sample_count}.png"))

                patches = outputs[-1][:1]
                # Convert from CHW to HWC
                patches = patches.permute(0, 2, 3, 1)
                pca_map = get_pca_map(patches, (512, 512))
                image = Image.fromarray((pca_map*255.).astype(np.uint8))
                image.save(f"debug/pca_map_{self.sample_count}.png")

        self.sample_count += 1

        return outputs[0] if len(outputs) == 1 else outputs

        images = center_padding(images, self.patch_size)
        img_h, img_w = images.shape[-2:]
        out_hw = (img_h // self.patch_size, img_w // self.patch_size)

        # clip stuff
        x = self.visual.conv1(images)
        x_hw = x.shape[-2:]
        x = E.rearrange(x, "b c h w -> b (h w) c")

        # concat cls token
        _cls_embed = E.repeat(self.visual.class_embedding, "c -> b 1 c", b=x.shape[0])
        x = torch.cat([_cls_embed.to(x.dtype), x], dim=1)

        # add pos embed
        pos_embed = resize_pos_embed(self.visual.positional_embedding, x_hw)
        x = self.visual.ln_pre(x + pos_embed.to(x.dtype))

        embeds = []
        for i, blk in enumerate(self.visual.transformer.resblocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for i, _x in enumerate(embeds):
            _x = tokens_to_output(self.output, _x[:, 1:], _x[:, 0], out_hw)
            outputs.append(_x)

        return outputs[0] if len(outputs) == 1 else outputs
