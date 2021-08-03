import os


def get_persistence_path(saving_path, metadata) -> str:
    parent_directory = os.path.join(saving_path,
                                    f'P{metadata.patient}',
                                    f'I{metadata.interval}')
    filename = f'C{metadata.channel}.mat'

    return os.path.join(parent_directory, filename)
