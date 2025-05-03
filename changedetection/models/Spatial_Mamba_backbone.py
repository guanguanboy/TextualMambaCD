from MambaCD.classification_spatial.models.spatialmamba import SpatialMamba,SpatialMambaLayer

from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


class Backbone_SpatialMamba(SpatialMamba):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):

        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None) 
        kwargs.update(norm_layer=norm_layer)       

        # add norm ========================
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        # modify layer ========================
        def layer_forward(self: SpatialMambaLayer, x):
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

            y = None
            if self.downsample is not None:
                y = self.downsample(x)

            return x, y

        for l in self.layers:
            l.forward = partial(layer_forward, l)

        # del self.classifier ===================
        del self.head
        del self.avgpool
        del self.norm

        self.load_pretrained(pretrained, key=kwargs.get('key','model'))

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x

        return outs


