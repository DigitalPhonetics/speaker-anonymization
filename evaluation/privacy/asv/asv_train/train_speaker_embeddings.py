# This code is based on
# https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/train_speaker_embeddings.py
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

# Dataset prep (parsing Libri-train-clean-360 and annotation into csv files)
from .libri_prepare import prepare_libri  # noqa
from .seame_prepare import prepare_seame
from .asv_dataset import ASVDatasetGenerator


class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded


        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        for module in [self.modules.compute_features, self.modules.mean_var_norm,
                           self.modules.embedding_model, self.modules.classifier]:
            for p in module.parameters():
                p.requires_grad = True

        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
                name=epoch
            )


def _convert_to_yaml(overrides):
    # convert dict to yaml for overrides
    yaml_string = ""
    for key in overrides:
        yaml_string += str(key) +': ' +str(overrides[key]) + '\n'
    return yaml_string.strip()


def train_asv_speaker_embeddings(config_file, hparams_file, run_opts):
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)
    
    # Convert dict to yaml for overrides"""
    overrides = _convert_to_yaml(hparams_file)
    
    with open(config_file) as f:
        hparams = load_hyperpyyaml(f, overrides)

    prepare_data = prepare_seame if 'seame' in hparams['data_folder'] else prepare_libri

    run_on_main(
        prepare_data,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": ["train", "dev"],
            "split_ratio": [90, 10],
            "num_utt": hparams["num_utt"],
            "num_spk": hparams["num_spk"],
            "seg_dur": hparams["sentence_len"],
            "skip_prep": hparams["skip_prep"],
            "anon": hparams["anon"],
            "utt_selected_ways": hparams["utt_selected_ways"]
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    asv_dataset_gen = ASVDatasetGenerator(hparams)
    train_data, valid_data = asv_dataset_gen.dataio_prep()
    print(len(train_data), len(valid_data))
    
   
    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=config_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if "pretrainer" in hparams.keys() and hparams['pretrained_path'] != 'None':
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(run_opts["device"])

    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
