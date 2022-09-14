# This code is based on the descriptions in https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/processing/PLDA_LDA.py
from pathlib import Path
from speechbrain.processing.PLDA_LDA import PLDA, StatObject_SB, Ndx, fast_PLDA_scoring
import numpy as np
import torch

class PLDAModel:

    def __init__(self, train_embeddings, results_path: Path=None):
        self.mean, self.F, self.Sigma = None, None, None

        files_exist = False
        if results_path and results_path.exists():
            files_exist = self.load_parameters(results_path)
        if not files_exist:
            self._train_plda(train_embeddings)
            self.save_parameters(results_path)

    def compute_distance(self, enrollment_vectors, trial_vectors):
        enrol_vecs = enrollment_vectors.cpu().numpy()
        en_sets, en_s, en_stat0 = self._get_vector_stats(enrol_vecs, sg_tag='en')
        en_stat = StatObject_SB(modelset=en_sets, segset=en_sets, start=en_s, stop=en_s, stat0=en_stat0,
                                stat1=enrol_vecs)

        trial_vecs = trial_vectors.cpu().numpy()
        te_sets, te_s, te_stat0 = self._get_vector_stats(trial_vecs, sg_tag='te')
        te_stat = StatObject_SB(modelset=te_sets, segset=te_sets, start=te_s, stop=te_s, stat0=te_stat0,
                                stat1=trial_vecs)

        ndx = Ndx(models=en_sets, testsegs=te_sets)
        scores_plda = fast_PLDA_scoring(en_stat, te_stat, ndx, self.mean, self.F, self.Sigma)
        return scores_plda.scoremat

    def save_parameters(self, filename):
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.save(filename / 'plda_mean.npy', self.mean)
        np.save(filename / 'plda_F.npy', self.F)
        np.save(filename / 'plda_Sigma.npy', self.Sigma)

    def load_parameters(self, dir_path):
        existing_files = [x.name for x in dir_path.glob('*')]
        files_exist = True
        if 'plda_mean.npy' in existing_files:
            self.mean = np.load(dir_path / 'plda_mean.npy')
        else:
            files_exist = False

        if 'plda_F.npy' in existing_files:
            self.F = np.load(dir_path / 'plda_F.npy')
        else:
            files_exist = False

        if 'plda_Sigma.npy' in existing_files:
            self.Sigma = np.load(dir_path / 'plda_Sigma.npy')
        else:
            files_exist = False
        return files_exist

    def _train_plda(self, train_embeddings):
        vectors = train_embeddings.speaker_vectors.to(torch.float64)
        speakers = train_embeddings.speakers

        modelset = np.array([f'md{speaker}' for speaker in speakers], dtype="|O")
        segset, s, stat0 = self._get_vector_stats(vectors, sg_tag='sg')

        xvectors_stat = StatObject_SB(modelset=modelset, segset=segset, start=s, stop=s, stat0=stat0,
                                      stat1=vectors.cpu().numpy())

        plda = PLDA(rank_f=100)
        plda.plda(xvectors_stat)

        self.mean = plda.mean
        self.F = plda.F
        self.Sigma = plda.Sigma

    def _get_vector_stats(self, vectors, sg_tag='sg'):
        N, dim = vectors.shape
        segset = np.array([f'{sg_tag}{i}' for i in range(N)], dtype="|O")
        s = np.array([None] * N)
        stat0 = np.array([[1.0]] * N)
        return segset, s, stat0