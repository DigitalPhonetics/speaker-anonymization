# This code is based on
# https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/voxceleb_prepare.py
import csv
import logging
import random
from pathlib import Path
import sys  # noqa F401
import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm.contrib import tqdm
from collections import defaultdict
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_libri_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
ENROL_CSV = "enrol.csv"
SAMPLERATE = 16000

def prepare_seame(
    data_folder,
    save_folder,
    splits=["train", "dev"],
    split_ratio=[90, 10],
    seg_dur=3.0,
    amp_th=5e-04,
    num_utt=None,
    num_spk=None,
    random_segment=False,
    skip_prep=False,
    anon = False,
    utt_selected_ways="spk-random",
):
    """
    Prepares the csv files for the libri datasets.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original libri  dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        txt file containing the verification split.
    splits : list
        List of splits to prepare from ['train', 'dev']
    split_ratio : list
        List if int for train and validation splits
    seg_dur : int
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).
    amp_th : float
        removes segments whose average amplitude is below the
        given threshold.
    source : str
        Path to the folder where the VoxCeleb dataset source is stored.
    num_utt: float
        How many utterances for each speaker used for training
    num_spk: float
        How many speakers used for training
    random_segment : bool
        Train random segments
    skip_prep: Bool
        If True, skip preparation.

    Example
    -------
    >>> from libri_prepare import prepare_libri
    >>> data_folder = 'LibriSpeech/train-clean-360/'
    >>> save_folder = 'libri/'
    >>> splits = ['train', 'dev']
    >>> split_ratio = [90, 10]
    >>> prepare_voxceleb(data_folder, save_folder, splits, split_ratio)
    """

    if skip_prep:
        return
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "split_ratio": split_ratio,
        "save_folder": save_folder,
        "seg_dur": seg_dur,
        "num_utt": num_utt,
        "num_spk": num_spk,
    }

    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)

    # Setting ouput files
    save_opt = save_folder / OPT_FILE
    save_csv_train = save_folder / TRAIN_CSV
    save_csv_dev = save_folder / DEV_CSV


    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    data_folder = [Path(data_folder)]

    # _check_voxceleb1_folders(data_folder, splits)

    msg = "\tCreating csv file for the Seame Dataset.."
    logger.info(msg)

    # Split data into 90% train and 10% validation (verification split)
    wav_lst_train, wav_lst_dev = _get_utt_split_lists(
        data_folder, split_ratio, num_utt, num_spk, anon, utt_selected_ways
    )

    # Creating csv file for training data
    if "train" in splits:
        prepare_csv(
            seg_dur, wav_lst_train, save_csv_train, random_segment, amp_th
        )

    if "dev" in splits:
        prepare_csv(seg_dur, wav_lst_dev, save_csv_dev, random_segment, amp_th)


    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, str(save_opt))


def skip(splits, save_folder, conf):
    """
    Detects if the voxceleb data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
    }
    for split in splits:
        if not Path(save_folder, split_files[split]).is_file():
            skip = False
    #  Checking saved options
    save_opt = save_folder / OPT_FILE
    if skip is True:
        if save_opt.is_file():
            opts_old = load_pkl(str(save_opt))
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip

# Used for verification split
def _get_utt_split_lists(
    data_folders, split_ratio, num_utt='ALL', num_spk='ALL', anon=False, utt_selected_ways="spk-random"
):
    """
    Tot. number of speakers libri-360=921
    Splits the audio file list into train and dev.
    """
    train_lst = []
    dev_lst = []

    logger.debug("Getting file list...")
    logger.info(f'{data_folders}')
    for data_folder in data_folders:
        spk_files = defaultdict(set)
        full_utt = 0
        with open(f'{data_folder}/wav.scp', 'r') as f:
            for line in f:
                utt, wav_path = line.split()
                temp = Path(wav_path).stem
                spk_id = temp.split('_')[0]
                spk_files[spk_id].add(wav_path)
                full_utt += 1

        logger.info(f'{len(spk_files)}')
        logger.debug("use all speakers and all utterances for training")

        # per speaker, use 90% of utterances for train and 10% for dev
        # we are doing this to make sure that we see all speakers during training

        dev_utterances = []
        train_utterances = []
        train_dev_splits = {}

        for spk, utterance_set in spk_files.items():
            utterance_list = list(utterance_set)
            random.shuffle(utterance_list)
            split = int(0.01 * split_ratio[0] * len(utterance_list))
            if split > 2:
                train_utterances.extend(utterance_list[:split])
                dev_utterances.extend(utterance_list[split:])
                train_dev_splits[spk] = {'train': len(utterance_list[:split]), 'dev': len(utterance_list[split:])}
            else:
                train_utterances.extend(utterance_list)
                train_dev_splits[spk] = {'train': len(utterance_list), 'dev': 0}

        train_lst.extend(train_utterances)
        dev_lst.extend(dev_utterances)

    return train_lst, dev_lst


def _get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks
    """
    num_chunks = int(audio_duration / seg_dur)  # all in milliseconds

    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]

    return chunk_lst


def prepare_csv(seg_dur, wav_lst, csv_file, random_segment=False, amp_th=0):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    random_segment: bool
        Read random segments
    amp_th: float
        Threshold on the average amplitude on the chunk.
        If under this threshold, the chunk is discarded.

    Returns
    -------
    None
    """

    msg = f'\t"Creating csv lists in  {csv_file}..."'
    logger.info(msg)

    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    # For assigning unique ID to each chunk
    my_sep = "--"
    entry = []
    problematic_wavs = []
    spks = set()
    avoided = 0
    # Processing all the wav files in the list
    for wav_file in tqdm(wav_lst, dynamic_ncols=True):
        # Getting sentence and speaker ids
        try:
            temp = wav_file.split("/")[-1].split(".")[0]
            spk_id = temp.split('_')[0]
            audio_id = temp.replace('_', my_sep)
        except ValueError:
            logger.info(f"Malformed path: {wav_file}")
            continue

        # Reading the signal (to retrieve duration in seconds)
        try:
            audio_duration = sf.info(wav_file).duration
        except RuntimeError:
            problematic_wavs.append(wav_file)
            continue

        if random_segment:
            start_sample = 0
            stop_sample = int(audio_duration * SAMPLERATE)

            # Composition of the csv_line
            csv_line = [
                audio_id,
                str(audio_duration),
                wav_file,
                start_sample,
                stop_sample,
                spk_id,
            ]
            entry.append(csv_line)
            spks.add(spk_id)
        else:
            #audio_duration = signal.shape[0] / SAMPLERATE
            signal, fs = torchaudio.load(wav_file)
            signal = signal.squeeze(0)

            uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]
                start_sample = int(float(s) * SAMPLERATE)
                end_sample = int(float(e) * SAMPLERATE)

                #  Avoid chunks with very small energy
                mean_sig = torch.mean(np.abs(signal[start_sample:end_sample]))
                if mean_sig < amp_th:
                    avoided += 1
                    continue

                # Composition of the csv_line
                csv_line = [
                    chunk,
                    str(audio_duration),
                    wav_file,
                    start_sample,
                    end_sample,
                    spk_id,
                ]
                entry.append(csv_line)
                spks.add(spk_id)

    print(avoided)

    logger.info(f'Skipped {len(problematic_wavs)} invalid audios')
    csv_output = csv_output + entry

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    # Final prints
    msg = f"\t{csv_file} successfully created!"
