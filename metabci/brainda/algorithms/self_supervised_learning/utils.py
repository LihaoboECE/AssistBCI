import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal import detrend
import os
from scipy import interpolate
from scipy import signal


def plot_embedding(data, label, title=None, auto_save=False, path=''):
    c = ['#ff2e63','#252a34','#08d9d6']
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], '.',
                 # color=plt.cm.Set2(label[i] +1),
                 color=c[label[i]])
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.xlim([np.min(data[:, 0]), np.max(data[:, 0])])
        plt.ylim([np.min(data[:, 1]), np.max(data[:, 1])])
        plt.title(title)

    if auto_save:
        num = len(os.listdir(path))
        plt.savefig(path+str(num))
    else:
        plt.show()

    return fig


'''
data augment function
Author: Li Haobo
Email: lihaoboece@gmail.com
'''
def augment_data(original_data, original_label=None, noise_scale=1.2, alpha=1.2,
                 n_segments=10, m_segments=10, distortion_factor_low=0.9,
                 distortion_factor_high=1.1, max_shift=5):
    """
    Apply multiple data augmentation methods to EEG data including channel manipulations

    Parameters:
        original_data: numpy array with shape (N, C, S)
        original_label: corresponding labels
        noise_scale: standard deviation of Gaussian noise
        alpha: scaling factor for amplitude transformation
        n_segments: number of segments for temporal dislocation
        m_segments: number of segments for time warping
        distortion_factor_low: time distortion compression factor
        distortion_factor_high: time distortion stretch factor
        max_shift: maximum point of signal channel shifting

    Returns:
        merged_data: augmented dataset
        acc_original: original data repeated to match augmented size
        label: labels repeated to match augmented size
    """
    N, C, S = original_data.shape

    # (1) Adding Gaussian noise
    noisy = original_data + np.random.normal(0, noise_scale, original_data.shape)
    #
    # (2) Scale transformation
    scaled = original_data * alpha

    # (3) Horizontal flipping
    h_flipped = -original_data

    # (4) Vertical flipping (time reversal)
    v_flipped = original_data[:, :, ::-1]

    # (5) Temporal dislocation
    dislocated = np.zeros_like(original_data)
    for i in range(N):
        segments = np.array_split(original_data[i], n_segments, axis=1)
        np.random.shuffle(segments)
        dislocated[i] = np.concatenate(segments, axis=1)

    # (6) Time warping
    time_warped = np.zeros_like(original_data)
    for i in range(N):
        segments = np.array_split(original_data[i], m_segments, axis=1)
        warped_segs = []
        for seg in segments:
            scale = np.random.uniform(distortion_factor_low, distortion_factor_high)
            orig_length = seg.shape[1]
            new_length = int(orig_length * scale)

            x_orig = np.linspace(0, 1, orig_length)
            x_new = np.linspace(0, 1, new_length)
            interp_seg = np.array([interpolate.interp1d(x_orig, ch)(x_new)
                                   for ch in seg])
            warped_segs.append(interp_seg)

        combined = np.concatenate(warped_segs, axis=1)
        time_warped[i] = np.array([signal.resample(ch, S) for ch in combined])

    # # (7) Channel swapping #good
    # channel_swapped = original_data.copy()
    # for i in range(N):
    #     # Randomly permute channels
    #     np.random.shuffle(channel_swapped[i])
    #
    # # (8) Channel-wise temporal shifting
    # channel_shifted = np.zeros_like(original_data)
    # for i in range(N):
    #     for c in range(C):
    #         shift_amount = np.random.randint(-max_shift, max_shift)
    #         if shift_amount > 0:
    #             # Shift forward
    #             channel_shifted[i, c, :-shift_amount] = original_data[i, c, shift_amount:]
    #             channel_shifted[i, c, -shift_amount:] = original_data[i, c, -1]
    #         elif shift_amount < 0:
    #             # Shift backward
    #             shift_amount = abs(shift_amount)
    #             channel_shifted[i, c, shift_amount:] = original_data[i, c, :-shift_amount]
    #             channel_shifted[i, c, :shift_amount] = original_data[i, c, 0]
    #         else:
    #             channel_shifted[i, c] = original_data[i, c]

    # Combine all augmented datasets
    augmented_list = [
        noisy, scaled, h_flipped, v_flipped,
        dislocated, time_warped
    ]
    # augmented_list = [
    #         channel_shifted
    #     ]
    # augmented_list = [
    #     original_data, dislocated, time_warped, channel_swapped, channel_shifted
    # ]
    #
    merged = np.concatenate(augmented_list, axis=0)

    # merged = np.stack((noisy, dislocated, channel_swapped, channel_shifted), axis=1)
    # merged = merged.reshape(-1, *merged.shape[2:])
    #
    acc_original = np.repeat(original_data, len(augmented_list), axis=0)

    if original_label is None:
        return merged, acc_original
    else:
        label = np.repeat(original_label, len(augmented_list), axis=0)
        return merged, acc_original, label



