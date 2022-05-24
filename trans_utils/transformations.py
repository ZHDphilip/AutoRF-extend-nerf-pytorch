import numpy as np
import math
import pickle


# in the code for 3D bbox detection, it is assumed that pitch and roll ~ 0
# this is valid for cars on the road in most cases
# reference code repo: https://github.com/skhadem/3D-BoundingBox
def euler_to_rotMat(yaw, pitch=0, roll=0):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]])
    # R = RzRyRx
    rot_Mat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    return rot_Mat

def get_transform_Mat(dx, dy, dz, yaw, pitch=0, roll=0):
    rot_Mat = euler_to_rotMat(yaw, pitch, roll) # 3x3
    translation_Vec = np.array([[dx], [dy], [dz]])
    proj_Mat = np.append(rot_Mat, translation_Vec, axis=1)
    proj_Mat = np.append(proj_Mat, [[0,0,0,1]], axis=0)
    return proj_Mat


if __name__ == '__main__':
    # sin, cos = 0.22091809, -0.9752923
    # yaw = math.atan2(sin, cos)  # ALWAYS USE THIS
    # yaw *= 180 / math.pi
    # if yaw < 0: yaw += 360
    # print(yaw)
    # # print(math.sin(yaw), math.cos(yaw))
    # print(math.sin(yaw*math.pi/180), math.cos(yaw*math.pi/180))
    # dx, dy, dz = 34.74307768278042, -1.6061516278674617, 66.79337301259723
    # Mat = get_transform_Mat(dx=dx, dy=dy, dz=dz, yaw=yaw)
    # print(Mat)
    pickle_file = open("./res.pickle", 'rb')
    log = pickle.load(pickle_file)
    data_dict = dict()
    for img_id, data in log.items():
        for i in range(len(data['predictions'])):
            tmp = dict()
            print(img_id)
            print(data['predictions'][i])
            pan_name = img_id.replace(".png", "_"+str(data['predictions'][i]['id'])+".png")
            print(pan_name)
            mask_name = img_id.replace(".png", "_"+str(data['predictions'][i]['id'])+"mask.png")
            print(mask_name)
            tmp['mask'] = mask_name
            sin, cos = data['predictions'][i]['orient']# [0], data['predictions'][i]['orient'][1]
            yaw = math.atan2(sin, cos)
            yaw *= 180 / math.pi
            if yaw < 0: yaw += 360
            dx, dy, dz = data['predictions'][i]['pose']
            Mat = get_transform_Mat(dx=dx, dy=dy, dz=dz, yaw=yaw)
            tmp['transform'] = Mat
            data_dict[pan_name] = tmp
            print(data_dict)
            exit()