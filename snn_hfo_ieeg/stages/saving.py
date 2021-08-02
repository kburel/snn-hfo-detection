import os
import scipy.io as sio


def save_hfo_detection(user_facing_hfo_detection, saving_path, metadata):
    parent_directory = os.path.join(saving_path,
                                    f'P{metadata.patient}',
                                    f'I{metadata.interval}')
    os.makedirs(parent_directory, exist_ok=True)
    filename = f'C{metadata.channel}.mat'

    filepath = os.path.join(parent_directory, filename)
    print(user_facing_hfo_detection._asdict())
    sio.savemat(filepath, user_facing_hfo_detection._asdict())
