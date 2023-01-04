# Speaker Anonymization

This repository contains the speaker anonymization system developed at the Institute for Natural Language Processing 
(IMS) at the University of Stuttgart, Germany. The system is described in the following papers:

| Paper | Published at | Branch | Demo |
|-------|--------------|--------|------|
| [Speaker Anonymization with Phonetic Intermediate Representations](https://www.isca-speech.org/archive/interspeech_2022/meyer22b_interspeech.html) | [Interspeech 2022](https://www.interspeech2022.org/) | [phonetic_representations](https://github.com/DigitalPhonetics/speaker-anonymization/tree/phonetic_representations) | [https://huggingface.co/spaces/sarinam/speaker-anonymization](https://huggingface.co/spaces/sarinam/speaker-anonymization) |
| [Anonymizing Speech with Generative Adversarial Networks to Preserve Speaker Privacy](https://arxiv.org/abs/2210.07002) | Soon at [SLT 2022](https://slt2022.org/) | [gan_embeddings](https://github.com/DigitalPhonetics/speaker-anonymization/tree/gan_embeddings) | coming soon |

If you want to see the code to the respective papers, go to the branch referenced in the table. The latest version 
of our system can be found here on the main branch.

**Check out our live demo on Hugging Face: [https://huggingface.co/spaces/sarinam/speaker-anonymization](https://huggingface.co/spaces/sarinam/speaker-anonymization)**

**Check also out [our contribution](https://www.voiceprivacychallenge.org/results-2022/docs/3___T04.pdf) to the [Voice Privacy Challenge 2022](https://www.voiceprivacychallenge.org/results-2022/)!**


## System Description
The system is based on the Voice Privacy Challenge 2020 which is included as submodule. It uses the basic idea of 
speaker embedding anonymization with neural synthesis, and uses the data and evaluation framework of the challenge. 
For a detailed description of the system, please read our Interspeech paper linked above.

### Added Features
Since the publication of the first paper, some features have been added. The new structure of the pipeline and its 
capabilities contain:
* **GAN-based speaker anonymization**: We show in [this paper](https://arxiv.org/abs/2210.07002) that a Wasserstein 
  GAN can be trained to generate artificial speaker embeddings that resemble real ones but are not connected to any 
  known speaker -- in our opinion, a crucial condition for anonymization. The current GAN model in the latest 
  release v2.0 has been trained to generate a custom type of 128-dimensional speaker embeddings (included also in our 
  speech 
  synthesis toolkit [IMSToucan](https://github.com/DigitalPhonetics/IMS-Toucan)) instead of x-vectors or ECAPA-TDNN 
  embeddings.
* **Prosody cloning**: We now provide an option to transfer the original prosody to the anonymized audio via [prosody 
  cloning](https://arxiv.org/abs/2206.12229)! If you want to avoid an exact cloning but modify it slightly (but 
  randomly to avoid reversability), use the random offset thresholds. They are given as lower and upper threshold, 
  as an percentage in relation to the modification. For instance, if you give these thresholds as (80, 120), you 
  will modify the pitch and energy values of each phone by multiplying it with a random value between 80% and 120% 
  (leading to either weakening or amplifying the signal).
* **ASR**: Our ASR is now using a [Branchformer](https://arxiv.org/abs/2207.02971) encoder and includes word 
  boundaries and stress markers in its output.

![architecture](figures/architecture.png)

The current code on the main branch expects the models of release v2.0. If you want to use the pipeline as presented at 
Interspeech 2022, 
please go to 
the 
[phonetic_representations branch](https://github.com/DigitalPhonetics/speaker-anonymization/tree/phonetic_representations).

## Installation
### 1. Clone repository
Clone this repository with all its submodules:
```
git clone --recurse-submodules https://github.com/DigitalPhonetics/speaker-anonymization.git
``` 

### 2. Download models
Download the models [from the release page (v2.0)](https://github.com/DigitalPhonetics/speaker-anonymization/releases/tag/v2.0), unzip the folders and place them into a *models* 
folder as stated in the release notes. Make sure to not unzip the single ASR models, only the outer folder.
```
cd speaker-anonymization
mkdir models
cd models
for file in anonymization asr tts; do
    wget https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/${file}.zip
    unzip ${file}.zip
    rm ${file}.zip
done
```

### 3. Install challenge framework
In order to be able to use the framework of the Voice Privacy Challenge 2020 for evaluation, you need to install it 
first. According to [the challenge repository](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020), this should simply be
```
cd Voice-Privacy-Challenge-2020
./install.sh
```
However, on our systems, we had to make certain adjustments and also decided to use a more light-weight environment 
that minimizes unnecessary components. If you are interested, you can see our steps in 
[alternative_challenge_framework_installation.md](alternative_challenge_framework_installation.md). Just as a note: It is 
very possible that those would not directly work on your system and would need to be modified.

**Note: this step will download and install Kaldi, and might lead to complications. Additionally, make sure that you 
are running the install script on a device with access to GPUs and CUDA.**

### 4. Install requirements
Additionally, install the [requirements](requirements.txt) (in the base directory of this repository):
```
pip install -r requirements.txt
```

## Getting started
Before the actual execution of our pipeline system, you first need to download and prepare the challenge data and 
the evaluation models. For 
this, you will need a password provided by the organizers of the Voice Privacy Challenge. Please contact them (see 
information on [their repository](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020) or 
[website](https://www.voiceprivacychallenge.org/)) for 
this access.

You can do this by either

### a) Executing our lightweight scripts: 
This will only download and prepare the necessary models and datasets. Note that these scripts are simply extracts 
of the challenge run script.
```
cd setup_scripts
./run_download_data.sh
./run_prepare_data.sh
```

or by
### b) Executing the challenge run script:
This will download and prepare everything necessary AND run the baseline system of the Voice Privacy Challenge 2020. 
Note that you will need to have installed the whole framework by the challenge install script before.
```
cd Voice-Privacy-Challenge-2020/baseline
./run.sh
```

### Running the pipeline
The system pipeline controlled in [run_inference.py](run_inference.py). You can run it via
```
python run_inference.py --gpu <gpu_id>
```
with <gpu_id> being the ID of the GPU the code should be executed on. If this option is not specified, it will run 
on CPU (not recommended).

The script will anonymize the development and test data of LibriSpeech and VCTK in three steps:
1. ASR: Recognition of the linguistic content, output in form of text or phone sequences
2. Anonymization: Modification of speaker embeddings, output as torch vectors
3. TTS: Synthesis based on recognized transcription, extracted prosody and anonymized speaker embedding, output as 
   audio files (wav)

Each module produces intermediate results that are saved to disk. A module is only executed if previous intermediate 
results for dependent pipeline combination do not exist or if recomputation is forced. Otherwise, the previous 
results are loaded. Example: The ASR module is 
only executed if there are no transcriptions produced by exactly that ASR model. On the other hand, the TTS is 
executed if (a) the ASR was performed directly before (new transcriptions), and/or (b) the anonymization was 
performed directly before (new speaker embeddings), and/or (c) no TTS results exist for this combination of models.

If you want to change any settings, like the particular models or datasets, you can adjust the *settings* dictionary 
in [run_inference.py](run_inference.py). If you want to force recomputation for a specific module, add its tag to 
the *force_compute* list. 

Immediately after the anonymization pipeline terminates, the evaluation pipeline is started. It performs some 
preparation steps and then executes the evaluation part of the challenge run script (this extract can be found in 
[evaluation/run_evaluation.sh](../speaker-anonymization/evaluation/run_evaluation.sh)).

Finally, for clarity, the most important parts of the evaluation results as well as the used settings are copied to 
the [results](results) directory.


## Citation
```
@inproceedings{meyer22b_interspeech,
  author={Sarina Meyer and Florian Lux and Pavel Denisov and Julia Koch and Pascal Tilli and Ngoc Thang Vu},
  title={{Speaker Anonymization with Phonetic Intermediate Representations}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={4925--4929},
  doi={10.21437/Interspeech.2022-10703}
}
```
