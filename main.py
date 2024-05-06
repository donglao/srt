# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Some parts are taken from https://github.com/Liusifei/UVC
"""
import argparse

import cv2
import torch
import torch.nn as nn

import utils
import vision_transformer as vits
from eval_utils import eval_segmentation
from distill_utils import distill
from extract_feature_offline import extract_davis_features
from feature_extractor import BasicExtractor, EnsembledExtractor
from datasets import DAVISDinofeaturesDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generation of ensembled features followed by distillation')
    # Model args
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    # Data args
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--output_dir', default=".", help='Path where to save segmentations')
    parser.add_argument('--data_path', default='/path/to/davis/', type=str)
    parser.add_argument('--data_split', default='train', type=str)
    parser.add_argument("--n_last_frames", type=int, default=7, help="number of preceeding frames")
    parser.add_argument("--size_mask_neighborhood", default=12, type=int,
        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")
    # Feature extraction args
    parser.add_argument("--do_extract", action='store_true')
    parser.add_argument("--feature_save_dir", type=str, default="data/features", help="path to save the extracted features")
    # Ensemble args
    parser.add_argument("--feature_extractor", type=str, default="basic", help="feature extractor to use")
    parser.add_argument("--dx", type=int, default=3, help="window size (width)")
    parser.add_argument("--dy", type=int, default=3, help="window size (height)")
    parser.add_argument("--bs", type=int, default=32, help="batch size for forward pass")
    # Distillation args
    parser.add_argument("--do_distill", action='store_true')
    parser.add_argument("--epochs", type=int, default=50, help="epochs for distillation")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate for distillation")
    parser.add_argument("--lr_decay", type=float, default=0.1, help="learning rate decay")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--sched_type", type=str, default='step', help="step/exp")
    parser.add_argument("--steps", type=str, default='25;40', help="steps separated by semicolon")
    parser.add_argument("--model_save_dir", type=str, default='', help="where to save distilled model")
    parser.add_argument("--scale_factor", type=int, default=1, help="scale factor for super-resolving the embeddings, only applies to FeatureExtractorFine")
    
    parser.add_argument("--unfreeze", type=str, default='all', help="what to unfreeze for distill")
    parser.add_argument("--loss", type=str, default='mse', help="mse/mae")

    # Evaluation args
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model_path", type=str, default='', help="where to load distilled model")
    args = parser.parse_args()

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # building network
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    if args.load_model_path != '':
        model.load_state_dict(torch.load(args.load_model_path))

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    assert model.patch_embed.patch_size == args.patch_size

    feature_extractor = utils.get_feature_extractor(args)

    if args.do_extract:
        extract_davis_features(model=model, 
                               extractor=feature_extractor,
                               davis_data_dir=args.data_path,
                               feature_save_dir=args.feature_save_dir)

    # Dataset code here (either load or create on the fly)
    dataset = DAVISDinofeaturesDataset(args.data_path, args.data_split)

    if args.do_distill:
        # Distillation code here
        model = distill(args, model, dataset)
        model.eval()
    
    if args.do_eval:
        eval_segmentation(args, model)

