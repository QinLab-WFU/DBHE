import os

import torchvision

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from argparse import Namespace

import timm
import torch
import torch.nn.functional as F
from torch import nn
from hyptorch.nn import ToPoincare


def build_model(args: Namespace, pretrained):
    last = (
        ToPoincare(
            c=args.hyp_c,
            ball_dim=args.n_bits,
            riemannian=False,
            clip_r=args.clip_r,
        )
        if args.hyp_c > 0
        else NormLayer()
    )

    if args.backbone == "vit-s":
        body = timm.create_model("vit_small_patch16_224", pretrained=pretrained)

        head = nn.Sequential(nn.Linear(body.head.in_features, args.n_bits), last)
        nn.init.constant_(head[0].bias.data, 0)
        nn.init.orthogonal_(head[0].weight.data)

        rm_head(body)

        if args.freeze is not None:
            freeze(body, args.freeze)

        model = HeadSwitch(body, head)

    elif args.backbone == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        body = torchvision.models.resnet50(weights=weights)

        head = nn.Sequential(nn.Linear(body.fc.in_features, args.n_bits), last)
        nn.init.constant_(head[0].bias.data, 0)
        nn.init.orthogonal_(head[0].weight.data)

        rm_head(body)

        if args.freeze is not None:
            for module in filter(lambda m: isinstance(m, nn.BatchNorm2d), body.modules()):
                module.eval()
                module.train = lambda _: None

        model = HeadSwitch(body, head)
    else:
        raise NotImplementedError(f"not supported backbone: {args.backbone}")

    return model.to(args.device)


class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


def freeze(model, num_block):
    def fr(m):
        for param in m.parameters():
            param.requires_grad = False

    fr(model.patch_embed)
    fr(model.pos_drop)
    for i in range(num_block):
        fr(model.blocks[i])


def rm_head(m):
    names = set(x[0] for x in m.named_children())
    target = {"head", "fc", "head_dist"}
    for x in names & target:
        m.add_module(x, nn.Identity())


class HeadSwitch(nn.Module):
    def __init__(self, body, head):
        super().__init__()
        self.body = body
        self.head = head
        self.norm = NormLayer()

    def forward(self, x, skip_poincare=True):
        x = self.body(x)
        # if isinstance(x, tuple):
        #     x = x[0]
        if skip_poincare:
            # x = self.head[0](x)  # <- add this!
            x = self.norm(x)
        else:
            x = self.head(x)
        return x


if __name__ == "__main__":
    image = torch.rand(2, 3, 224, 224).to("cuda:0")
    model = build_model(
        Namespace(
            backbone="vit-s",
            # backbone="resnet50",
            n_bits=16,
            hyp_c=0.1,
            clip_r=2.3,
            freeze=0,
            device="cuda:0",
        ),
        True,
    )
    # print(model)
    output = model(image, skip_poincare=True)
    print(output.shape)
