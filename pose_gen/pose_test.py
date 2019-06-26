import pickle
import numpy as np
import cv2
import glob
import os
import sys
from config import PHEONIX_FOLDER, OPENPOSE_FOLDER

import cv2

sys.path.append(os.path.join(OPENPOSE_FOLDER, "build/python"))
from openpose import pyopenpose as op


class PoseEstimator():
    def __init__(self, hand=True, face=True):
        params = dict()
        params["model_folder"] = os.path.join(OPENPOSE_FOLDER, "models")
        params["face"] = hand
        params["hand"] = face
        params["scale_number"] = 4
        params["hand_scale_number"] = 6
        params["hand_scale_range"] = 0.4
        params["scale_gap"] = 0.25
        params["keypoint_scale"] = 3
        params["num_gpu"] = 1
        # params["model_pose"] = "COCO"
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

    @staticmethod
    def image_with_pose(pose_file, img, new=False):
        with open(pose_file, 'rb') as f:
            pose_data = pickle.load(f)

            body = pose_data['body']

            if body.shape[0] > 1 or body.shape[0] < 1:
                return False

            body = body.squeeze()

            lhand = pose_data['left_hand'].squeeze()
            rhand = pose_data['right_hand'].squeeze()

            pose = np.vstack((body, lhand, rhand))
            face = pose_data['face'].squeeze()
            pose = np.vstack((face, pose))

            pose = pose[:, :2]
            if new:
                pose *= np.array([210, 260])

            pose = pose.astype(np.int32)
            for point in pose:
                cv2.circle(img, (point[0], point[1]), 2, (0, 255, 0), -1)

            return True


if __name__ == "__main__":
    datapath = "/home/kenny/Workspace/Data/SLT/02December_2011_Friday_tagesschau-8010/"
    ls = glob.glob(datapath + "/*_pose.pkl")
    # ls = glob.glob(datapath + "/*.png")
    # pose_estim = PoseEstimator()
    # for image_file in ls:
    #     datum = pose_estim.estimate_pose(image_file)
    #     if datum is None: continue
    #     pose_filename = image_file.replace(".png", "_pose_new.pkl")
    #     with open(pose_filename, 'wb') as f:
    #         pose_data = {"body": datum.poseKeypoints,
    #                      "face": datum.faceKeypoints,
    #                      "left_hand": datum.handKeypoints[0],
    #                      "right_hand": datum.handKeypoints[1]}
    #
    #         pickle.dump(pose_data, f)

    for pose_file_old in ls:
        image_file_new = pose_file_old.replace("_pose.pkl", ".png")
        image_file_old = image_file_new.replace("8010", "8010_old")
        img_new = cv2.imread(image_file_new)
        img_new_raw = cv2.imread(image_file_new)
        img_old = cv2.imread(image_file_old)
        img_old_raw = cv2.imread(image_file_old)
        if not isinstance(img_new, np.ndarray):
            continue
        pose_file_new = pose_file_old.replace("_pose.pkl", "_pose_new.pkl")

        if not PoseEstimator.image_with_pose(pose_file_old, img_old): continue
        if not PoseEstimator.image_with_pose(pose_file_new, img_new, new=True): continue

        cv2.imshow("New with pose", cv2.resize(img_new, (0, 0), fx=2, fy=2))
        cv2.imshow("Old with pose", cv2.resize(img_old, (0, 0), fx=2, fy=2))
        cv2.imshow("New without pose", cv2.resize(img_new_raw, (0, 0), fx=2, fy=2))
        cv2.imshow("Old without pose", cv2.resize(img_old_raw, (0, 0), fx=2, fy=2))

        if cv2.waitKey(0) == 27:
            break
