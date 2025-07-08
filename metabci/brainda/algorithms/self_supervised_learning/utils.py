import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal import detrend
import os
from scipy import interpolate
from scipy import signal

def plot_point_cov(points, nstd=3, ax=None, **kwargs):
    # 求所有点的均值作为置信圆的圆心
    pos = points.mean(axis=0)
    # 求协方差
    cov = np.cov(points, rowvar=False)

    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=3, ax=None, **kwargs):
    def eigsorted(cov):
        cov = np.array(cov)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip


#画置信圆
def show_ellipse(X_pca, y, pca, regions=None):
    # 定义颜色

    colors = ['tab:blue', 'tab:orange', 'seagreen']

    if not regions:
        regions = ['Class 0', 'Class 1', 'Class 2']

    # 定义分辨率
    plt.figure(dpi=100, figsize=(40, 20))
    # 三分类则为3
    for i in range(0, 3):
        pts = X_pca[y == int(i), :]
        new_x, new_y = X_pca[y == i, 0], X_pca[y == i, 1]

        plt.plot(new_x, new_y, '.', color=colors[i], label=regions[i], markersize=14)

        plot_point_cov(pts, nstd=3, alpha=0.25, color=colors[i])

    # 添加坐标轴
    # plt.xlim(-3.5, 4.5)
    # plt.ylim(-1.5, 1.7)
    # plt.xticks(size=16, family='Times New Roman')
    # plt.yticks(size=16, family='Times New Roman')
    # font = {'family': 'Times New Roman', 'size': 16}
    plt.xticks(size=16)
    plt.yticks(size=16)
    font = { 'size': 16}

    plt.xlabel('PC1 ({} %)'.format(round(pca.explained_variance_ratio_[0] * 100, 2)), font)
    plt.ylabel('PC2 ({} %)'.format(round(pca.explained_variance_ratio_[1] * 100, 2)), font)

    # plt.legend(prop={"family": "Times New Roman", "size": 9}, loc='upper right')
    plt.legend(prop={"size": 9}, loc='upper right')
    plt.show()


def plot_embedding(data, label, title, auto_save=False):
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
    # plt.xlim([np.min(data[:, 0]), np.max(data[:, 0])])
    # plt.ylim([np.min(data[:, 1]), np.max(data[:, 1])])
    # plt.title(title)

    if auto_save:
        path = '/home/ubuntu-user/DC227998/2025EmotionSystemChannel amplification/models/img/LH_FT/'
        num = len(os.listdir(path))
        plt.savefig(path+str(num))
    else:
        plt.show()

    return fig



'''
PSD, DE, time domain features extraction
'''
def compute_psd_de(data, window, fs, f_bands=None, LDS=False, lds_window=10, step=None):
    """
    compute  DE (differential entropy) and PSD (power spectral density) features

    input:
	data-[n, m] n channels, m points of each time course,
	window-integer, window lens of each segment in seconds, such as 1s
	fs-integer, frequency of singal sampling rate, such as 200Hz
	optional  f_bands, default delta, theta, aplha, beta, gamma

    output:
        psd,de  [bands, channels, features]
        [channels, windows, bands]
    """
    if step==None:
        step = window
    channels, lens = data.shape
    data = detrend(data, axis=-1)

    segment_lens = window * fs
    step = int(step * fs)
    # samples = lens // segment_lens
    # index = samples * segment_lens
    # data = data[:, :index].contiguous()
    new_data = []
    i = 0
    while i + segment_lens <= lens:
        new_data.append(data[:, i:i + segment_lens])
        i += step
    data = np.stack(new_data, axis=1) #channels * windows * samples
    index = i

    if f_bands == None:
        f_bands = [(1, 4), (4, 7), (8, 13), (14, 30), (31, 49)]  # delta, theta, aplha, beta, gamma

    hamming_window = np.hamming(data.shape[-1])
    data = data * hamming_window

    # compute the magnitudes
    fxx = np.fft.fft(data)
    # fxx = fxx/fxx.shape[-1]

    timestep = 1 / fs
    f = np.fft.fftfreq(segment_lens, timestep)[:segment_lens // 2]  # only use the positive frequency
    fxx = np.abs(fxx[:, :, :segment_lens // 2])

    psd_bands = []
    de_bands = []
    for f_band1, f_band2 in f_bands:
        f_mask = (f >= f_band1) & (f <= f_band2)
        data_bands = fxx[:, :, f_mask]

        # psd = np.sum(data_bands ** 2 / (segment_lens // 2),
        #              axis=-1)  # same with scipy.signal.periodogram * fs, divide the number of total frequency bands like 100
        psd = np.mean(data_bands**2, axis=-1)  # only divide the number of frequency band1-band2 like 1-4, maybe 4 points with window==1s or 7 points with window==2s
        # de = np.log(2 * np.pi * np.exp(1) * (data_bands.var(axis=-1) ** 2)) / 2   #channels*segments
        de = np.log2(100*psd) # channels*windows

        psd_bands.append(psd)
        de_bands.append(de)

    psd = np.stack(psd_bands, axis=2)
    de = np.stack(de_bands, axis=2)

    # Remove singleton dimensions if any
    psd = psd.squeeze()
    de = de.squeeze()

    other_feature = []
    for segment in data:
        other_feature.append(extract_features(segment))
    other_feature = np.stack(other_feature, axis=0)


    # psd, de, other_feature = zscore(psd, axis=-1), zscore(de, axis=-1),zscore(other_feature, axis=-1)
    return psd, de, other_feature

    # features = [psd, de, other_feature]
    #
    # features_lds = []
    # for feature in features:
    #     #features: [channels, windows, bands]
    #     feature = uniform_filter1d(feature, size=3, axis=1, mode='nearest')
    #     feature = feature.transpose(2, 0, 1) #features: [bands, channels, windows]
    #     lds_all = []
    #     for band in feature:
    #         lds_channels = []
    #         for channel in band:
    #             para = {}
    #             para['u0'] = np.array([np.mean(channel)])[np.newaxis, :]
    #             para['V0'] = np.array([0.01])[np.newaxis, :]
    #             para['A'] = np.array([1.0])[np.newaxis, :]
    #             para['T'] = np.array([0.0001])[np.newaxis, :]
    #             para['C'] = np.array([1.0])[np.newaxis, :]
    #             para['sigma'] = np.array([1.0])[np.newaxis, :]
    #             para['givenAll'] = np.array([1.0])[np.newaxis, :]
    #             channel =channel[np.newaxis, :]
    #             F_lds = dlm_inference(channel, para=para)
    #             lds_channels.append(F_lds['z'].squeeze())
    #         lds_channels = np.stack(lds_channels, axis=0)
    #         lds_all.append(lds_channels)
    #     lds_all = np.stack(lds_all, axis=0)
    #     lds_all = lds_all.transpose(1, 2, 0)
    #     features_lds.append(lds_all)
    #
    # psd, de, other_feature = features_lds
    #
    # return psd, de, other_feature
    # return psd, de, index


def extract_features(X, m_sampen=2, r_factor_sampen=0.2, num_bins_shannon=10):
    num_signals, num_samples = X.shape

    # Power
    power = np.mean(X ** 2, axis=1)

    # Line Length
    line_length = np.sum(np.abs(np.diff(X, axis=1)), axis=1)

    # RMS
    rms = np.sqrt(power)

    # First Difference
    first_diff = np.mean(np.abs(np.diff(X, axis=1)), axis=1)

    # Second Difference
    second_diff = np.mean(np.abs(np.diff(X, n=2, axis=1)), axis=1)

    # # Spectral Entropy
    # spectral_entropy_feat = np.zeros(num_signals)
    # for i in range(num_signals):
    #     signal = X[i, :]
    #     fft_vals = np.fft.rfft(signal)
    #     psd = np.abs(fft_vals) ** 2
    #     psd_sum = np.sum(psd) + 1e-12  # Avoid division by zero
    #     psd_norm = psd / psd_sum
    #     psd_norm = psd_norm[psd_norm > 0]  # Remove zeros to avoid log(0)
    #     spectral_entropy_feat[i] = -np.sum(psd_norm * np.log(psd_norm))
    #
    # # Shannon Entropy
    # shannon_entropy_feat = np.zeros(num_signals)
    # for i in range(num_signals):
    #     signal = X[i, :]
    #     hist, _ = np.histogram(signal, bins=num_bins_shannon)
    #     hist = hist.astype(float)
    #     prob = hist / hist.sum()
    #     prob = prob[prob > 0]
    #     shannon_entropy_feat[i] = -np.sum(prob * np.log2(prob))

    # # Sample Entropy
    # sample_entropy_feat = np.zeros(num_signals)
    # for i in range(num_signals):
    #     signal = X[i, :]
    #     N = len(signal)
    #     if N <= m_sampen:
    #         sample_entropy_feat[i] = 0.0
    #         continue
    #     # Compute r as a factor of the signal's std
    #     r = r_factor_sampen * np.std(signal)
    #     # Generate templates for m and m+1
    #     templates_m = [signal[j:j + m_sampen] for j in range(N - m_sampen)]
    #     if len(templates_m) < 2:
    #         sample_entropy_feat[i] = 0.0
    #         continue
    #     # Count matches for m
    #     A = 0
    #     for j in range(len(templates_m)):
    #         for k in range(len(templates_m)):
    #             if j != k:
    #                 if np.max(np.abs(np.subtract(templates_m[j], templates_m[k]))) <= r:
    #                     A += 1
    #     # Generate templates for m+1
    #     templates_m1 = [signal[j:j + m_sampen + 1] for j in range(N - m_sampen - 1)]
    #     if len(templates_m1) < 2:
    #         sample_entropy_feat[i] = 0.0
    #         continue
    #     # Count matches for m+1
    #     B = 0
    #     for j in range(len(templates_m1)):
    #         for k in range(len(templates_m1)):
    #             if j != k:
    #                 if np.max(np.abs(np.subtract(templates_m1[j], templates_m1[k]))) <= r:
    #                     B += 1
    #     # Compute sample entropy
    #     if A == 0 or B == 0:
    #         sample_entropy_feat[i] = 0.0
    #     else:
    #         sample_entropy_feat[i] = -np.log(B / A)

    # Combine all features into a 2D array
    features = np.vstack([
        power,
        line_length,
        rms,
        first_diff,
        second_diff,
        # spectral_entropy_feat,
        # shannon_entropy_feat,
        # sample_entropy_feat
    ]).T

    return features


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



