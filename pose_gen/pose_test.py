import pickle
import numpy as np
import cv2
import glob
import re

if __name__ == "__main__":
    datapath = "/home/kenny/Workspace/Data/SLT/open_pose/fullFrame-227x227px/test/01April_2010_Thursday_tagesschau-4330/"

    ls = glob.glob(datapath + "/*.pkl")
    ls = sorted(list(ls), key= lambda x: )
    for pose_file in ls:
        image_file = pose_file.replace("_pose.pkl", ".png")
        img = cv2.imread(image_file)

        with open(pose_file, 'rb') as f:
            pose_data = pickle.load(f)

            body_idxs = list(range(9)) + list(range(15, 19))
            body = pose_data['body']

            if body.shape[0] > 1 or body.shape[0] < 1:
                continue

            body = body.squeeze()[body_idxs]
            lhand = pose_data['left_hand'].squeeze()
            rhand = pose_data['right_hand'].squeeze()

            pose = np.vstack((body, lhand, rhand))
            face = pose_data['face'].squeeze()
            pose = np.vstack((face, pose))
            pose = pose[:, :2].astype(np.int32)

            for point in pose:
                cv2.circle(img, (point[0], point[1]), 3, (0,255,0), -1)

        cv2.imshow("window", img)



        if cv2.waitKey(0) == 27:
            break
