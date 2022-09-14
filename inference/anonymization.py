import json
import numpy as np
import torch
from sklearn.preprocessing import minmax_scale, StandardScaler

from anonymization import PoolAnonymizer, RandomAnonymizer
from utils import create_clean_dir


ANON_MODELS = {
    'pool': PoolAnonymizer,
    'random': RandomAnonymizer
}


class InferenceAnonymizer:

    def __init__(self, model_name, data_dir, results_dir, model_dir, vectors_dir, device, force_compute=False):
        self.force_compute = force_compute
        self.results_dir = results_dir / 'speaker_embeddings' / model_name
        self.data_dir = data_dir
        self.vectors_dir = vectors_dir
        self.device = device
        self.scaling = None
        self.std_scaler = None

        self.dim_ranges = self._load_dim_ranges(model_dir / 'anonymization' / model_name)
        self.anonymizer = self._load_anonymizer(model_dir / 'anonymization' / model_name)

    def anonymize_embeddings(self, dataset):
        dataset_results_dir = self.results_dir / dataset
        if dataset_results_dir.exists() and any(dataset_results_dir.iterdir()) and not self.force_compute:
            # if there are already anonymized speaker embeddings from this model and the computation is not forces,
            # simply load them
            print('No computation of anonymized embeddings necessary; load existing anonymized speaker embeddings '
                  'instead...')
            anon_embeddings = self.anonymizer.load_embeddings(dataset_results_dir)
            return anon_embeddings, False
        else:
            # otherwise, create new anonymized speaker embeddings
            print('Anonymize speaker embeddings...')
            anon_embeddings = self.anonymizer.anonymize_data(self.data_dir / dataset,
                                                             vector_dir=self.vectors_dir / dataset)
            if self.dim_ranges:
                anon_embeddings = self._scale_embeddings(anon_embeddings)
            create_clean_dir(dataset_results_dir)  # deletes existing results files
            self.anonymizer.save_embeddings(anon_embeddings, dataset_results_dir)
            return anon_embeddings, True

    def _load_dim_ranges(self, model_dir):
        if (model_dir / 'stats_per_dim.json').exists():
            with open(model_dir / 'stats_per_dim.json') as f:
                dim_ranges = json.load(f)
                return [(v['min'], v['max']) for k, v in sorted(dim_ranges.items(), key=lambda x: int(x[0]))]

    def _load_anonymizer(self, model_dir):
        model_name = model_dir.name.lower()

        if 'pool' in model_name:
            model_type = 'pool'
        else:
            model_type = 'random'

        print(f'Model type of anonymizer: {model_type}')

        model = ANON_MODELS[model_type](device=self.device)
        model.load_parameters(model_dir)

        if 'minmax' in model_name:
            self.scaling = 'minmax'
        elif 'std_scale' in model_name and model_type == 'pool':
            self.scaling = 'std'
            self.std_scaler = StandardScaler()
            self.std_scaler.fit(model.pool_embeddings.speaker_vectors.cpu().numpy())

        return model

    def _scale_embeddings(self, embeddings):
        vectors = embeddings.speaker_vectors.cpu().numpy()

        if self.scaling == 'minmax':
            scaled_dims = []
            for i in range(len(self.dim_ranges)):
                scaled_dims.append(minmax_scale(vectors[:, i], self.dim_ranges[i], axis=0))

            scaled_vectors = torch.tensor(np.array(scaled_dims)).T.to(self.device)
            embeddings.speaker_vectors = scaled_vectors
        elif self.scaling == 'std':
            scaled_vectors = torch.tensor(self.std_scaler.transform(vectors))
            embeddings.speaker_vectors = scaled_vectors
        return embeddings


