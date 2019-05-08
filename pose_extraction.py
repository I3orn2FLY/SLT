from config import PHEONIX_FOLDER, OPENPOSE_FOLDER
import os
import sys
import glob
import numpy as np
import pickle
# You need to import cv2 first
import cv2

sys.path.append(os.path.join(OPENPOSE_FOLDER, "build/python"))
from openpose import pyopenpose as op

pose_data_folder = os.path.join(PHEONIX_FOLDER, "open_pose")
images_folder = os.path.join(PHEONIX_FOLDER, "features")


class PoseEstimator():
    def __init__(self, hand=True, face=True):
        params = dict()
        params["model_folder"] = os.path.join(OPENPOSE_FOLDER, "models")
        params["face"] = hand
        params["hand"] = face
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def estimate_pose(self, filename):
        datum = op.Datum()
        imageToProcess = cv2.imread(filename)
        if not isinstance(imageToProcess, np.ndarray):
            return None
        datum.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop([datum])

        return datum


def create_folders(image_files):
    for idx, filename in enumerate(image_files):
        filename = os.path.join(pose_data_folder, filename)
        path = os.path.split(filename)[0]
        if not os.path.exists(path):
            os.makedirs(path)

        if idx % 10000 == 0:
            print("\r %d" % idx, end=" ")


def create_image_files():
    ls = glob.iglob(images_folder + "/**/*.*", recursive=True)
    with open('images_file.txt', 'w') as f:
        for idx, filename in enumerate(ls):
            shortened_name = filename.split("features/")[1]
            f.write(shortened_name + "\n")

    with open('current_idx.txt', 'w') as f:
        f.write(str(-1))


if __name__ == "__main__":

    folders_created = True
    image_files_created = True

    if len(sys.argv) > 1:
        pose_data_folder = pose_data_folder.replace("kenny", "nurobot427")
        images_folder = images_folder.replace("kenny", "nurobot427")

    if not image_files_created: create_image_files()

    with open('image_files.txt', 'r') as f:
        image_files = [x.strip() for x in f.readlines()]

    if not folders_created: create_folders(image_files)

    L = len(image_files)

    with open('current_idx.txt', 'r') as f:
        cur_idx = int(f.readline())

    pose_estim = PoseEstimator()

    for idx, filename in enumerate(image_files[cur_idx + 1:]):
        idx = idx + cur_idx + 1
        image_filename = os.path.join(images_folder, filename)
        pose_filename = os.path.splitext(os.path.join(pose_data_folder, filename))[0] + "_pose.pkl"
        datum = pose_estim.estimate_pose(image_filename)
        if datum is None: continue
        with open(pose_filename, 'wb') as f:
            pose_data = {"body": datum.poseKeypoints,
                         "face": datum.faceKeypoints,
                         "left_hand": datum.handKeypoints[0],
                         "right_hand": datum.handKeypoints[1]}

            pickle.dump(pose_data, f)

        if idx % 100 == 0:
            with open('current_idx.txt', 'w') as f:
                f.write(str(idx))

            print("\r %.2f" % (idx * 100 / L), end=" ")

    print()
