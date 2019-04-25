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

if __name__ == "__main__":
    pose_data_folder = os.path.join(PHEONIX_FOLDER, "open_pose")
    images_folder = os.path.join(PHEONIX_FOLDER, "features")

    ls = glob.iglob(images_folder + "/**/*.*", recursive=True)

    create_folders = False

    if create_folders:
        for idx, filename in enumerate(ls):
            filename = filename.replace(images_folder, pose_data_folder)
            path = os.path.split(filename)[0]
            if not os.path.exists(path):
                os.makedirs(path)

            if idx % 10000 == 0:
                print("\r %d" % idx, end=" ")

    L = 1890000



    with open('completed_images.txt', 'r') as f:
        completed_images = f.readlines()

    completed_images = [x.strip() for x in completed_images]

    completed_images_file = open('completed_images.txt', 'a')

    params = dict()
    params["model_folder"] = os.path.join(OPENPOSE_FOLDER, "models")
    params["face"] = True
    params["hand"] = True

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    try:

        for idx, filename in enumerate(ls):
            if filename in completed_images:
                continue

            datum = op.Datum()
            imageToProcess = cv2.imread(filename)
            if not isinstance(imageToProcess, np.ndarray):
                continue
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])

            file, ext = os.path.splitext(filename.replace(images_folder, pose_data_folder))
            pose_file = file + "_pose.pkl"


            with open(pose_file, 'wb') as f:
                pose_data = {"body": datum.poseKeypoints,
                             "face": datum.faceKeypoints,
                             "left_hand": datum.handKeypoints[0],
                             "right_hand": datum.handKeypoints[1]}

                pickle.dump(pose_data, f)


            completed_images_file.write(filename + "\n")
            if idx % 100 == 0:
                print("\r %.2f" % (idx * 100 / L), end=" ")

    except:
        completed_images_file.close()

    print()
