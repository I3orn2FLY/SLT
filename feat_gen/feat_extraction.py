from config import PHEONIX_FOLDER, OPENPOSE_FOLDER
import pandas as pd
import os
import glob
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import cv2

pose_data_folder = os.path.join(PHEONIX_FOLDER, "open_pose")
images_folder = os.path.join(PHEONIX_FOLDER, "features")


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_num_from_path(x):
    return int(x.split("/")[-1].split(".")[0][-4:])


class FeatureExtractor():

    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet50(pretrained=True)
        self.model.fc = Identity()
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_features(self, video_path, side=224):
        ls = sorted(list(glob.glob(video_path)), key=lambda x: get_num_from_path(x))
        imgs = []
        with torch.no_grad():
            for img_path in ls:
                img = cv2.resize(cv2.imread(img_path), (side, side))
                img = img.transpose([2, 0, 1]) / 225
                imgs.append(self.normalize(torch.Tensor(img)))

            inp = torch.stack(imgs).to(self.device)
            feats = self.model(inp).detach().cpu().numpy()

            return feats


def generate_split_data(feat_extractor, split="train", side=224,
                        out_data_path="/home/kenny/Workspace/Data/SLT/resnet_feats/"):

    out_data_path += str(side) + "x" + str(side) + "/"
    df = pd.read_csv(PHEONIX_FOLDER + "/annotations/manual/PHOENIX-2014-T." + split + ".corpus.csv", sep='|')
    print("Generating", split, "split features.")
    L = len(df)
    for idx, video_path in enumerate(df.video):
        video_path = PHEONIX_FOLDER + "/features/fullFrame-210x260px/" + split + "/" + video_path.replace("/1/", "/")

        feats = feat_extractor.get_features(video_path)

        feat_path = video_path.replace(PHEONIX_FOLDER + "/features/fullFrame-210x260px/", out_data_path)
        feat_path = os.path.split(feat_path)[0]
        if not os.path.exists(os.path.split(feat_path)[0]):
            os.makedirs(feat_path)
        np.save(feat_path, feats)

        percent = idx / (L - 1) * 100
        print("\rProgress %.2f" % percent, end=" ")

    print()


if __name__ == "__main__":
    feat_extractor = FeatureExtractor()
    generate_split_data(feat_extractor, split="test")
    generate_split_data(feat_extractor, split="dev")
