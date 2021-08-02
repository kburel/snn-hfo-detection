import scipy.io as sio


def save_hfo_detection(user_facing_sfo_detection, saving_path):
    sio.savemat(saving_path, user_facing_sfo_detection)
