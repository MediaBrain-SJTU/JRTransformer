import numpy as np

def batch_VIM(GT, pred, select_frames=[1, 3, 7, 9, 13]):
    '''Calculate the VIM at selected timestamps.

    Args:
        GT: [B, T, J, 3].
    
    Returns:
        errorPose: [T].
    '''
    errorPose = np.power(GT - pred, 2)
    errorPose = np.sum(errorPose, axis=(2, 3))
    errorPose = np.sqrt(errorPose)
    errorPose = errorPose.sum(axis=0)
    return errorPose[select_frames]


def batch_MPJPE(GT, pred, select_frames=[1, 3, 7, 9, 13]):
    '''Calculate the MPJPE at selected timestamps.

    Args:
        GT: [B, T, J, 3], np.array, ground-truth pose position in world coordinate system (meter).
        pred: [B, T, J, 3], np.array, predicted pose position.

    Returns:
        errorPose: [T], MPJPE at selected timestamps.
    '''

    errorPose = np.power(GT - pred, 2)
    # B, T, J, 3
    errorPose = np.sum(errorPose, -1)
    errorPose = np.sqrt(errorPose)
    # B, T, J
    errorPose = errorPose.sum(axis=-1) / pred.shape[2]
    # B, T
    errorPose = errorPose.sum(axis=0)
    # T
    return errorPose[select_frames]

def batch_VIM_(GT, pred, select_frames=[1, 3, 7, 9, 13]):
    '''Calculate the VIM at selected timestamps.

    Args:
        GT: [B, J, T, 3].
    
    Returns:
        errorPose: [T].
    '''
    errorPose = np.power(GT - pred, 2)
    errorPose = np.sum(errorPose, axis=(1, 3))
    errorPose = np.sqrt(errorPose)
    errorPose = errorPose.sum(axis=0)
    return errorPose[select_frames]

def batch_MPJPE_(GT, pred, select_frames=[1, 3, 7, 9, 13]):
    '''Calculate the MPJPE at selected timestamps.

    Args:
        GT: [B, J, T, 3], np.array, ground-truth pose position in world coordinate system (meter).
        pred: [B, J, T, 3], np.array, predicted pose position.

    Returns:
        errorPose: [T], MPJPE at selected timestamps.
    '''

    errorPose = np.power(GT - pred, 2)
    # B, J, T, 3
    errorPose = np.sum(errorPose, -1)
    errorPose = np.sqrt(errorPose)
    # B, J, T
    errorPose = errorPose.sum(axis=1) / pred.shape[1]
    # B, T
    errorPose = errorPose.sum(axis=0)
    # T
    return errorPose[select_frames]