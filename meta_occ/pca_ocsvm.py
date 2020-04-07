from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

import torch


class PCAOneClassSVM:
    def __init__(self, nu, gamma, var_retained, kernel='rbf'):
        self._pca = PCA(n_components=var_retained)
        self._svm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def infer(self, batch_support_inputs, batch_query_inputs):
        if isinstance(batch_support_inputs, torch.Tensor):
            batch_support_inputs = batch_support_inputs.cpu().detach().numpy()
        if isinstance(batch_query_inputs, torch.Tensor):
            batch_query_inputs = batch_query_inputs.cpu().detach().numpy()

        task_batch_size = len(batch_query_inputs)
        episodes = zip(batch_support_inputs, batch_query_inputs)
        preds = []
        for episode_support_inputs, episode_query_inputs in episodes:
            if episode_support_inputs.ndim > 2:
                episode_support_inputs = episode_support_inputs.reshape(
                    len(episode_support_inputs), -1)
            if episode_query_inputs.ndim > 2:
                episode_query_inputs = episode_query_inputs.reshape(
                    len(episode_query_inputs), -1)

            episode_support_features = self._pca.fit_transform(
                episode_support_inputs)
            episode_query_features = self._pca.transform(episode_query_inputs)

            self._svm.fit(episode_support_features)

            episode_preds = self._svm.predict(episode_query_features)
            preds.extend(episode_preds)

        preds = torch.tensor(preds)
        preds = preds.reshape(task_batch_size, -1)
        return preds
