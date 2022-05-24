import pickle
import numpy as np
import math
import imageio
from trans_utils.transformations import euler_to_rotMat, get_transform_Mat

intrinsic = [
    9.799200e+02, 0.000000e+00, 6.900000e+02,
    0.000000e+00, 9.741183e+02, 2.486443e+02,
    0.000000e+00, 0.000000e+00, 1.000000e+00
]


def load_custorm_dataset(basedir):
    pickle_file = open(basedir + "/res.pickle", 'rb')
    log = pickle.load(pickle_file)
    data_dict = dict()
    K = []
    H = []
    W = []
    count = 0
    imgs = []
    masks = []
    poses = []
    focal = intrinsic[0][0]
    for img_id, data in log.items():
        for i in range(len(data['predictions'])):
            pan_name = img_id.replace(".png", "_"+str(data['predictions'][i]['id'])+".png")
            img = imageio.imread(basedir+"/"+pan_name)
            H.append(img.shape[0])
            W.append(img.shape[1])
            k = intrinsic
            k[0][2], k[1][2] = H[-1]/2, W[-1]/2
            K.append(k)
            imgs.append(img)
            mask_name = img_id.replace(".png", "_"+str(data['predictions'][i]['id'])+"mask.png")
            masks.append(imageio.imread(basedir+"/"+mask_name))
            sin, cos = data['predictions'][i]['orient']# [0], data['predictions'][i]['orient'][1]
            yaw = math.atan2(sin, cos)
            yaw *= 180 / math.pi
            if yaw < 0: yaw += 360
            dx, dy, dz = data['predictions'][i]['pose']
            Mat = get_transform_Mat(dx=dx, dy=dy, dz=dz, yaw=yaw)
            poses.append(Mat)
            count += 1
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    masks = (np.array(masks) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    split = [0, 5*count//7, count//7, count - 5*count//7 -count//7]
    i_split = [np.arange(split[i], split[i+1]) for i in range(3)]
    return imgs, poses, masks, [H, W, focal], K, i_split
    