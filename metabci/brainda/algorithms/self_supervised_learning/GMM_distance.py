import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import cdist, mahalanobis
from collections import defaultdict


class PCA_classifier:
    def __init__(self, n_features, pca_patch_max_len=20, max_buf_size=50,
                 min_samples_per_class=15, adjustment_step=0.1, max_adjustment=10.0):
        self.n_features = int(n_features)
        self.buf_features = None
        self.buf_labels = None
        self.original_buf_features = None

        self.pca_components = n_features
        self.pca = PCA(n_components=self.pca_components)
        self.pca_patch = None
        self.pca_patch_max_len = pca_patch_max_len

        self.max_buf_size = max_buf_size
        self.min_samples_per_class = min_samples_per_class

        self.labels = None
        self.inclass_adjustment_history = defaultdict(list)
        self.outclass_adjustment_history = defaultdict(list)
        self.adjustment_step = adjustment_step
        self.max_adjustment = max_adjustment

        self.new_class_detection_state = defaultdict(lambda: {
            'count': 0,
            'last_detection': -1000,
            'diversity_score': 0.0
        })
        self.sample_counter = 0

        # 用于距离计算的属性
        self.means_ = None
        self.covariances_ = None
        self.inv_covariances_ = None  # 用于马氏距离的逆协方差矩阵

    def fit(self, x, y):
        if self.original_buf_features is None:
            self.original_buf_features, self.buf_labels = x, y
        else:
            self.original_buf_features = np.concatenate((self.original_buf_features, x), axis=0)
            self.buf_labels = np.concatenate((self.buf_labels, y), axis=0)

        if self.labels is None:
            self.labels = np.unique(y)
            self.n_components = self.labels.shape[0]
            # self.outclass_bias, self.inclass_bias = np.zeros((self.n_components,)), np.zeros((self.n_components,))
        elif not np.isin(y, self.labels).all():
            _new_labels = np.setdiff1d(y, self.labels)
            _old_labels = np.intersect1d(y, self.labels)

            # self.outclass_bias[np.isin(self.labels, _old_labels)] += 1
            # self.outclass_bias = np.concatenate((self.outclass_bias, np.zeros((_new_labels.shape[0]))), axis=0)
            # self.inclass_bias = np.concatenate((self.inclass_bias, np.zeros((_new_labels.shape[0]))), axis=0)

            self.labels = np.concatenate((self.labels, _new_labels), axis=0)
            self.n_components = self.labels.shape[0]
        else:
            _old_labels = np.intersect1d(y, self.labels)
            # self.outclass_bias[np.isin(self.labels, _old_labels)] += 1

        self.updata_model()

    def _clean_buffer_based_on_diversity(self):
        if len(self.buf_labels) <= self.max_buf_size:
            return

        print("start buffer cleaning")
        unique_labels = np.unique(self.buf_labels)
        keep_indices = []
        for label in unique_labels:
            label_indices = np.where(self.buf_labels == label)[0]
            label_features = self.buf_features[label_indices]

            min_keep = min(self.min_samples_per_class, len(label_indices))
            dist_matrix = cdist(label_features, label_features, metric='euclidean')
            np.fill_diagonal(dist_matrix, np.inf)

            selected = self._select_diverse_samples(
                features=label_features,
                dist_matrix=dist_matrix,
                min_keep=min_keep,
                max_keep=int(self.max_buf_size / len(self.buf_labels)))
            keep_indices.extend(label_indices[selected])

            self.buf_features = self.buf_features[keep_indices]
            self.buf_labels = self.buf_labels[keep_indices]
            self.original_buf_features = self.original_buf_features[keep_indices]

    def _select_diverse_samples(self, features, dist_matrix, min_keep, max_keep):
        n_samples = len(features)
        selected = set()
        avg_dists = np.mean(dist_matrix, axis=1)
        first_sample = np.argmax(avg_dists)
        selected.add(first_sample)

        while len(selected) < min_keep and len(selected) < n_samples:
            min_dists = np.min(dist_matrix[:, list(selected)], axis=1)
            min_dists[list(selected)] = -np.inf
            next_sample = np.argmax(min_dists)
            selected.add(next_sample)

        if len(selected) < max_keep:
            remaining = set(range(n_samples)) - selected
            additional = min(max_keep - len(selected), len(remaining))
            selected.update(sorted(remaining, reverse=True)[:additional])

        return sorted(selected)

    def updata_model(self):
        if self.pca_patch is None:
            self.buf_features = self.pca.fit_transform(self.original_buf_features)
        else:
            self.buf_features = self.pca.fit_transform(
                np.concatenate((self.original_buf_features, self.pca_patch), axis=0))
            self.buf_features = self.buf_features[:self.original_buf_features.shape[0], :]

        self._clean_buffer_based_on_diversity()

        # 初始化距离相关的属性
        self.threshold_in_class = np.zeros((self.n_components,))
        self.threshold_out_class = np.zeros((self.n_components,))
        self.means_ = [np.zeros((1, self.pca_components)) for _ in range(self.n_components)]
        self.covariances_ = [np.eye(self.pca_components) for _ in range(self.n_components)]
        self.inv_covariances_ = [np.eye(self.pca_components) for _ in range(self.n_components)]

        for k in range(self.n_components):
            _buf = self.buf_features[self.buf_labels == self.labels[k]]
            self.means_[k] = np.mean(_buf, axis=0)

            # 计算协方差矩阵及其逆矩阵
            if len(_buf) > 1:
                cov = LedoitWolf().fit(_buf).covariance_
                self.covariances_[k] = cov
                try:
                    self.inv_covariances_[k] = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    self.inv_covariances_[k] = np.linalg.pinv(cov)

            # 计算距离并确定阈值
            distances = self._calculate_distances(_buf, k)
            self.determine_outclass_threshold(distances, k)
            self.determine_inclass_threshold(distances, k)

    def _calculate_distances(self, samples, class_idx):
        """计算样本到类中心的距离（欧氏距离）"""
        center = self.means_[class_idx]
        return np.linalg.norm(samples - center, axis=1)

    def determine_outclass_threshold(self, distances, k):
        """基于距离确定类间阈值"""
        if len(distances) == 0:
            self.threshold_out_class[k] = np.inf
            return

        q1 = np.quantile(distances, 0.25)
        q3 = np.quantile(distances, 0.65)
        iqr = q3 - q1
        self.threshold_out_class[k] = q3

    def determine_inclass_threshold(self, distances, k):
        """基于距离确定类内阈值"""
        if len(distances) == 0:
            self.threshold_in_class[k] = 0
            return

        q1 = np.quantile(distances, 0.25)
        q3 = np.quantile(distances, 0.35)
        iqr = q3 - q1
        # self.threshold_in_class[k] = q1 - 1.5 * iqr
        self.threshold_in_class[k] = q3

        # 确保阈值不为负
        # self.threshold_in_class[k] = max(0, self.threshold_in_class[k])

    def error_pred_feedback(self, label):
        """错误预测反馈"""
        index = np.where(self.labels == label)[0][0]
        if self.inclass_adjustment_history[index]:
            avg_adjust = np.mean(self.inclass_adjustment_history[index][-3:])
            adjustment = np.clip(avg_adjust + self.adjustment_step,
                                 -self.max_adjustment, self.max_adjustment)
        else:
            adjustment = self.adjustment_step

        # 调整阈值
        self.threshold_in_class[index] *= (1 - adjustment)
        # self.threshold_out_class[index] *= (1 + adjustment)
        self.inclass_adjustment_history[index].append(adjustment)

    def predict(self, x):
        '''
        :param x:
        return: new_class, pred_classes, report
        '''
        if len(x.shape) == 1:
            x = x[np.newaxis, :]

        if self.pca_patch is None:
            self.pca_patch = x
        elif self.pca_patch.shape[0] < self.pca_patch_max_len:
            self.pca_patch = np.concatenate((self.pca_patch, x), axis=0)
        else:
            self.pca_patch = np.roll(self.pca_patch, -1)
            self.pca_patch[-1] = x

        self.updata_model()
        x = self.pca.transform(x)

        # 计算每个样本到各类中心的距离
        dists = np.zeros((x.shape[0], self.n_components))
        for i, sample in enumerate(x):
            for k in range(self.n_components):
                # 使用欧氏距离
                dists[i, k] = np.linalg.norm(sample - self.means_[k])

        # 初始化输出
        new_class = []
        pred_classes = []
        report = []

        for sample_dists in dists:
            # 检查是否是新类
            is_new_class = np.all(sample_dists > self.threshold_out_class)
            new_class.append(is_new_class)

            # 检查是否需要报告
            report_needed = np.any(sample_dists <= self.threshold_in_class)
            report.append(report_needed)

            # 获取预测类别
            if report_needed:
                # 报告模式：返回所有低于类内阈值的类别
                above_threshold = np.where(sample_dists <= self.threshold_in_class)[0]
                sorted_indices = above_threshold[np.argsort(sample_dists[above_threshold])]
                pred_classes.append([self.labels[idx] for idx in sorted_indices])
            else:
                # 非报告模式：返回距离最小的类别
                pred_class = np.argmin(sample_dists)
                pred_classes.append([self.labels[pred_class]])

        return new_class, pred_classes, report

    def should_trigger_new_class_detection(self, current_label, buffer, MIN_DETECTION_INTERVAL, MIN_ACC_NEW_CLASS):
        buffer = self.pca.transform(buffer)
        state = self.new_class_detection_state[current_label]
        sample_idx = self.sample_counter

        cool = not (sample_idx - state['last_detection'] < MIN_DETECTION_INTERVAL)
        diversity = self.calculate_diversity(buffer)
        state['diversity_score'] = diversity
        state['count'] += 1

        # 4. 置信度检查
        confidence = self.calculate_confidence(buffer, current_label)

        return (cool and state['count'] >= MIN_ACC_NEW_CLASS)
        # return (cool and state['count'] >= MIN_ACC_NEW_CLASS)

    def calculate_diversity(self, buffer):
        feature_vars = np.var(buffer, axis=0)
        avg_feature_var = np.mean(feature_vars)
        dist_matrix = cdist(buffer, buffer, metric='euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        min_dists = np.min(dist_matrix, axis=1)
        avg_min_dist = np.mean(min_dists)
        diversity = min(1.0, 0.5 * avg_feature_var + 0.5 * avg_min_dist)
        return diversity

    def calculate_confidence(self, buffer, current_label):
        """计算新类置信度"""
        # 1. 与已知类别的距离
        class_distances = []
        for k in range(self.n_components):
            if self.labels[k] == current_label:
                continue

            center = self.means_[k]
            dist = np.mean(cdist(buffer, center[np.newaxis, :]))
            class_distances.append(dist)

        # 2. 与当前类别的距离
        current_center = self.means_[np.where(self.labels == current_label)[0][0]]
        current_dist = np.mean(cdist(buffer, current_center[np.newaxis, :]))

        # 3. 置信度计算
        min_other_dist = min(class_distances) if class_distances else current_dist * 2
        confidence = min(1.0, (min_other_dist - current_dist) / min_other_dist)
        return confidence

    def reset_detection_state(self, current_label):
        state = self.new_class_detection_state[current_label]
        state['count'] = 0
        state['last_detection'] = self.sample_counter



class PCA_EmoAdapt_online:
    def __init__(self, backbone, n_features, device, original_buf_EEG=None, buf_labels=None,
                 pca_patch_max_len=20, max_buf_size=50,
                 min_samples_per_class=15, adjustment_step=0.1, max_adjustment=10.0):

        self.backbone = backbone.to(device)
        # self.device = device

        self.n_features = int(n_features)

        self.buf_labels = buf_labels # can be reload
        self.original_buf_EEG = original_buf_EEG # can be reload

        self.buf_features = None

        self.pca_components = n_features
        self.pca = PCA(n_components=self.pca_components)
        self.pca_patch = None
        self.pca_patch_max_len = pca_patch_max_len

        self.max_buf_size = max_buf_size
        self.min_samples_per_class = min_samples_per_class

        self.labels = None
        self.inclass_adjustment_history = defaultdict(list)
        self.outclass_adjustment_history = defaultdict(list)
        self.adjustment_step = adjustment_step
        self.max_adjustment = max_adjustment

        self.new_class_detection_state = defaultdict(lambda: {
            'count': 0,
            'last_detection': -1000,
            'diversity_score': 0.0
        })
        self.sample_counter = 0

        # 用于距离计算的属性
        self.means_ = None
        self.covariances_ = None
        self.inv_covariances_ = None  # 用于马氏距离的逆协方差矩阵

    def fit(self, x, y):
        if self.original_buf_EEG is None:
            self.original_buf_EEG, self.buf_labels = x, y
        else:
            self.original_buf_EEG = np.concatenate((self.original_buf_EEG, x), axis=0)
            self.buf_labels = np.concatenate((self.buf_labels, y), axis=0)

        if self.labels is None:
            self.labels = np.unique(y)
            self.n_components = self.labels.shape[0]
        elif not np.isin(y, self.labels).all():
            _new_labels = np.setdiff1d(y, self.labels)
            # _old_labels = np.intersect1d(y, self.labels)
            self.labels = np.concatenate((self.labels, _new_labels), axis=0)
            self.n_components = self.labels.shape[0]

        self.updata_model()


    def init_model(self):
        if not self.original_buf_EEG is None and not self.buf_labels is None:
            self.labels = np.unique(self.buf_labels)
            self.n_components = self.labels.shape[0]
            self.updata_model()
            return
        else:
            print("Fall to init classifier, with Empty data buffer.")
            return

    def _clean_buffer_based_on_diversity(self):
        if len(self.buf_labels) <= self.max_buf_size:
            return

        print("start buffer cleaning")
        unique_labels = np.unique(self.buf_labels)
        keep_indices = []
        for label in unique_labels:
            label_indices = np.where(self.buf_labels == label)[0]
            label_features = self.buf_features[label_indices]

            min_keep = min(self.min_samples_per_class, len(label_indices))
            dist_matrix = cdist(label_features, label_features, metric='euclidean')
            np.fill_diagonal(dist_matrix, np.inf)

            selected = self._select_diverse_samples(
                features=label_features,
                dist_matrix=dist_matrix,
                min_keep=min_keep,
                max_keep=int(self.max_buf_size / len(self.buf_labels)))
            keep_indices.extend(label_indices[selected])

            self.buf_features = self.buf_features[keep_indices]
            self.buf_labels = self.buf_labels[keep_indices]
            self.original_buf_EEG = self.original_buf_EEG[keep_indices]

    def _select_diverse_samples(self, features, dist_matrix, min_keep, max_keep):
        n_samples = len(features)
        selected = set()
        avg_dists = np.mean(dist_matrix, axis=1)
        first_sample = np.argmax(avg_dists)
        selected.add(first_sample)

        while len(selected) < min_keep and len(selected) < n_samples:
            min_dists = np.min(dist_matrix[:, list(selected)], axis=1)
            min_dists[list(selected)] = -np.inf
            next_sample = np.argmax(min_dists)
            selected.add(next_sample)

        if len(selected) < max_keep:
            remaining = set(range(n_samples)) - selected
            additional = min(max_keep - len(selected), len(remaining))
            selected.update(sorted(remaining, reverse=True)[:additional])

        return sorted(selected)

    def updata_model(self):
        if self.pca_patch is None:
            self.buf_features = self.pca.fit_transform(
                self.backbone.predict(self.original_buf_EEG, disable_BN=True)
            )
        else:
            self.buf_features = self.pca.fit_transform(
                self.backbone.predict(
                    np.concatenate((self.original_buf_EEG, self.pca_patch), axis=0),
                    disable_BN=True
                )
            )
            self.buf_features = self.buf_features[:self.original_buf_EEG.shape[0], :]

        # self._clean_buffer_based_on_diversity()

        # 初始化距离相关的属性
        self.threshold_in_class = np.zeros((self.n_components,))
        self.threshold_out_class = np.zeros((self.n_components,))
        self.means_ = [np.zeros((1, self.pca_components)) for _ in range(self.n_components)]
        self.covariances_ = [np.eye(self.pca_components) for _ in range(self.n_components)]
        self.inv_covariances_ = [np.eye(self.pca_components) for _ in range(self.n_components)]

        for k in range(self.n_components):
            _buf = self.buf_features[self.buf_labels == self.labels[k]]
            self.means_[k] = np.mean(_buf, axis=0)

            # 计算协方差矩阵及其逆矩阵
            if len(_buf) > 1:
                cov = LedoitWolf().fit(_buf).covariance_
                self.covariances_[k] = cov
                try:
                    self.inv_covariances_[k] = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    self.inv_covariances_[k] = np.linalg.pinv(cov)

            # 计算距离并确定阈值
            distances = self._calculate_distances(_buf, k)
            self.determine_outclass_threshold(distances, k)
            self.determine_inclass_threshold(distances, k)

    def _calculate_distances(self, samples, class_idx):
        """计算样本到类中心的距离（欧氏距离）"""
        center = self.means_[class_idx]
        return np.linalg.norm(samples - center, axis=1)

    def determine_outclass_threshold(self, distances, k):
        """基于距离确定类间阈值"""
        # if len(distances) == 0:
        #     self.threshold_out_class[k] = np.inf
        #     return

        q1 = np.quantile(distances, 0.25)
        q3 = np.quantile(distances, 0.65) # for seed
        iqr = q3 - q1
        self.threshold_out_class[k] = q3

    def determine_inclass_threshold(self, distances, k):
        """基于距离确定类内阈值"""
        # if len(distances) == 0:
        #     self.threshold_in_class[k] = 0
        #     return

        q1 = np.quantile(distances, 0.25)
        q3 = np.quantile(distances, 0.35)
        iqr = q3 - q1
        # self.threshold_in_class[k] = q1 - 1.5 * iqr
        self.threshold_in_class[k] = q3

        # 确保阈值不为负
        # self.threshold_in_class[k] = max(0, self.threshold_in_class[k])

    def error_pred_feedback(self, label):
        """错误预测反馈"""
        index = np.where(self.labels == label)[0][0]
        if self.inclass_adjustment_history[index]:
            avg_adjust = np.mean(self.inclass_adjustment_history[index][-3:])
            adjustment = np.clip(avg_adjust + self.adjustment_step,
                                 -self.max_adjustment, self.max_adjustment)
        else:
            adjustment = self.adjustment_step

        # 调整阈值
        self.threshold_in_class[index] *= (1 - adjustment)
        # self.threshold_out_class[index] *= (1 + adjustment)
        self.inclass_adjustment_history[index].append(adjustment)

    def predict(self, x):
        '''
        :param x:
        return: new_class, pred_classes, report
        '''
        if len(x.shape) == 1:
            x = x[np.newaxis, :]

        if self.pca_patch is None:
            self.pca_patch = x
        elif self.pca_patch.shape[0] < self.pca_patch_max_len:
            self.pca_patch = np.concatenate((self.pca_patch, x), axis=0)
        else:
            self.pca_patch = np.roll(self.pca_patch, -1)
            self.pca_patch[-1] = x

        self.updata_model()
        x = self.pca.transform(self.backbone.predict(x, disable_BN=True))

        # 计算每个样本到各类中心的距离
        dists = np.zeros((x.shape[0], self.n_components))
        for i, sample in enumerate(x):
            for k in range(self.n_components):
                # 使用欧氏距离
                dists[i, k] = np.linalg.norm(sample - self.means_[k])

        # 初始化输出
        new_class = []
        pred_classes = []
        report = []

        for sample_dists in dists:
            # 检查是否是新类
            is_new_class = np.all(sample_dists > self.threshold_out_class)
            new_class.append(is_new_class)

            # 检查是否需要报告
            report_needed = np.any(sample_dists <= self.threshold_in_class)
            report.append(report_needed)

            # 获取预测类别
            if report_needed:
                # 报告模式：返回所有低于类内阈值的类别
                above_threshold = np.where(sample_dists <= self.threshold_in_class)[0]
                sorted_indices = above_threshold[np.argsort(sample_dists[above_threshold])]
                pred_classes.append([self.labels[idx] for idx in sorted_indices])
            else:
                # 非报告模式：返回距离最小的类别
                pred_class = np.argmin(sample_dists)
                pred_classes.append([self.labels[pred_class]])

        return new_class, pred_classes, report

    def should_trigger_new_class_detection(self, current_label, buffer, MIN_DETECTION_INTERVAL, MIN_ACC_NEW_CLASS):
        buffer = self.pca.transform(self.backbone.predict(buffer, disable_BN=True))
        state = self.new_class_detection_state[current_label]
        sample_idx = self.sample_counter

        cool = sample_idx - state['last_detection'] > MIN_DETECTION_INTERVAL
        diversity = self.calculate_diversity(buffer)
        state['diversity_score'] = diversity
        state['count'] += 1

        # 4. 置信度检查
        confidence = self.calculate_confidence(buffer, current_label)

        return (cool and state['count'] >= MIN_ACC_NEW_CLASS)
        # return (cool and state['count'] >= MIN_ACC_NEW_CLASS)

    def calculate_diversity(self, buffer):
        feature_vars = np.var(buffer, axis=0)
        avg_feature_var = np.mean(feature_vars)
        dist_matrix = cdist(buffer, buffer, metric='euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        min_dists = np.min(dist_matrix, axis=1)
        avg_min_dist = np.mean(min_dists)
        diversity = min(1.0, 0.5 * avg_feature_var + 0.5 * avg_min_dist)
        return diversity

    def calculate_confidence(self, buffer, current_label):
        """计算新类置信度"""
        # 1. 与已知类别的距离
        class_distances = []
        for k in range(self.n_components):
            if self.labels[k] == current_label:
                continue

            center = self.means_[k]
            dist = np.mean(cdist(buffer, center[np.newaxis, :]))
            class_distances.append(dist)

        # 2. 与当前类别的距离
        current_center = self.means_[np.where(self.labels == current_label)[0][0]]
        current_dist = np.mean(cdist(buffer, current_center[np.newaxis, :]))

        # 3. 置信度计算
        min_other_dist = min(class_distances) if class_distances else current_dist * 2
        confidence = min(1.0, (min_other_dist - current_dist) / min_other_dist)
        return confidence

    def reset_detection_state(self, current_label):
        state = self.new_class_detection_state[current_label]
        state['count'] = 0
        state['last_detection'] = self.sample_counter