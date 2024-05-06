import torch
from torchvision import transforms
import numpy as np
import os, glob, tqdm
from PIL import Image
import cv2

def read_frame(frame_dir, scale_size=[480]):
    """
    read a single frame & preprocess
    """
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if(ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img, ori_h, ori_w

def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


def extract_davis_features(model, extractor, davis_data_dir, feature_save_dir):
    davis_image_dir = os.path.join(davis_data_dir, "JPEGImages", "480p")

    davis_image_paths = glob.glob(os.path.join(davis_image_dir, "*", "*.jpg"))

    for image_path in tqdm.tqdm(davis_image_paths):
        image, _, _ = read_frame(image_path)

        feat = extractor.extract_feature(model, image)

        feat = feat.cpu().detach().numpy()
        save_path = os.path.join(feature_save_dir, *image_path.split("/")[-2:])
        save_path = save_path.replace(".jpg", ".npy")
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        np.save(save_path, feat)
