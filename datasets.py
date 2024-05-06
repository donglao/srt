import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import os, glob
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


class DAVISDinofeaturesDataset(Dataset):
    def __init__(self, data_dir, split):
        super(DAVISDinofeaturesDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split

        self.video_list_file = os.path.join(data_dir, "DAVIS", "ImageSets", "2017", "{}.txt".format(split))
        self.video_list = open(self.video_list_file).readlines()
        self.video_list = [a.strip() for a in self.video_list]

        self.features_dir = "{}-features".format(data_dir)

        self.image_dir = os.path.join(data_dir, "DAVIS", "JPEGImages", "480p")

        self.transform = transforms.Compose([
            transforms.Resize(480),# TODO: check how to resize here
            # transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.len = 0
        self.video_frame_list = []
        for video_name in self.video_list:
            video_frames = glob.glob(os.path.join(self.image_dir, video_name, "*.jpg"))
            self.video_frame_list += video_frames
            self.len += len(video_frames)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        video_frame_file = self.video_frame_list[index]
        frame_feature_file = os.path.join(self.features_dir, *video_frame_file.split("/")[-2:]).replace(".jpg", ".npy")

        video_frame = read_frame(video_frame_file)[0]
        frame_feature = np.load(frame_feature_file)

        video_frame = self.transform(video_frame)
        frame_feature = torch.Tensor(frame_feature)

        return video_frame, frame_feature
