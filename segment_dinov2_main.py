import argparse
import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F

from utils import get_feature_extractor
import sys
sys.path.insert(0, "external_src/dinov2")

import dinov2.eval.segmentation.models
from eval_ade20k import eval_ade20k
from PIL import Image
import urllib
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor
import matplotlib
from torchvision import transforms


EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

def create_segmenter(cfg, backbone_model):
    model = init_segmentor(cfg)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    model.init_weights()
    return model


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")

def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])


def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)


def main():
    parser = argparse.ArgumentParser('Generation of ensembled features for depth features')
    parser.add_argument('--arch', default='small', type=str,
        choices=['small', 'base', 'large', 'giant'], help='Architecture')
    # parser.add_argument('--head', default='dpt', type=str,
    #     choices=['dpt', 'linear', 'linear4'], help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    # parser.add_argument('--data_path', help="path to the nyu_v2 data")
    parser.add_argument('--do_eval', action='store_true', help='whether to perform evaluation')
    parser.add_argument('--head_type', choices=['linear', 'ms'], default='linear', help='prediction head to use')
    parser.add_argument("--head_dataset", choices=["ade20k", "voc2012"], default="ade20k", help="prediction head trained on with dataset")
    parser.add_argument("--head_scale_count", type=int, choices=[1,2,3,4,5], default=3, help="more scales: slower but better results, in (1,2,3,4,5)")
    # Ensemble args
    parser.add_argument("--feature_extractor", type=str, default=None, help="feature extractor to use")
    parser.add_argument("--dx", type=int, default=3, help="window size (width)")
    parser.add_argument("--dy", type=int, default=3, help="window size (height)")
    parser.add_argument("--bs", type=int, default=32, help="batch size for forward pass")
    # segmentation args
    parser.add_argument("--show_dir", type=str, default=None, help="If specified, the results will be dumped into the directory to save output results.")
    parser.add_argument("--metric_output", type=str, required=True, help="Path to save the evaluation metrics")
    args = parser.parse_args()

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[args.arch]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.cuda()

    HEAD_SCALE_COUNT = args.head_scale_count # more scales: slower but better results, in (1,2,3,4,5)
    HEAD_DATASET = args.head_dataset # in ("ade20k", "voc2012")
    HEAD_TYPE = args.head_type # in ("ms, "linear")


    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    print(cfg)
    if HEAD_TYPE == "ms":
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
        print("scales:", cfg.data.test.pipeline[1]["img_ratios"])
    print(cfg)
    model = create_segmenter(cfg, backbone_model=backbone_model)
    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    model.cuda()
    model.eval()

    # image = load_image_from_url(EXAMPLE_IMAGE_URL)

    # transform = make_depth_transform()

    # scale_factor = 1
    # rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
    # transformed_image = transform(rescaled_image)
    # batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image

    extractor = get_feature_extractor(args)

    if args.do_eval:
        eval_ade20k(args, model, extractor)

    # with torch.inference_mode():
    #     assert len(batch) == 1
    #     result = extractor.extract_feature(model, batch[0])

    # print(f"Output depth: {result.shape}")
    # depth_image = render_depth(result.squeeze().cpu())

if __name__ == '__main__':
    main()
