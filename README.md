# Speaker Anonymization

This repository contains the speaker anonymization system developed at the Institute for Natural Language Processing 
(IMS) at the University of Stuttgart, Germany.

This branch contains the code for our paper [Anonymizing Speech with Generative Adversarial Networks to Preserve 
Speaker Privacy](https://arxiv.org/abs/2210.07002) which we will present soon at the [SLT 2022](https://slt2022.org/)
. It is an extension of our first anonymization system described in [Speaker Anonymization with Phonetic 
Intermediate Representations](https://www.isca-speech.org/archive/interspeech_2022/meyer22b_interspeech.html) but 
uses a Wasserstein Generative Adversarial Network to generate artificial target speakers. 

If you want to see a list of all papers and implementations within this project, please visit the [main branch](https://github.com/DigitalPhonetics/speaker-anonymization/tree/main).

[comment]: <> (**Check out the live demo to this code on Hugging Face: [https://huggingface.co/spaces/sarinam/speaker-anonymization]&#40;https://huggingface.co/spaces/sarinam/speaker-anonymization&#41;**)

This implementation is similar to [our submission](https://www.voiceprivacychallenge.org/results-2022/docs/3___T04.pdf)
to the 
[Voice Privacy Challenge 2022](https://www.voiceprivacychallenge.org/results-2022/).


## System Description
The system is based on the Voice Privacy Challenge 2020 which is included as submodule. It uses the basic idea of 
speaker embedding anonymization with neural synthesis, and uses the data and evaluation framework of the challenge. 
For detailed descriptions of the system, please read our papers linked above.

### Added Features
The system is an extension of our first anonymization pipeline described in [our Interspeech paper](https://www.isca-speech.org/archive/interspeech_2022/meyer22b_interspeech.html).
Given an input audio, two kinds of information are extracted: (a) the linguistic content in form of phone sequences 
using a custom Speech Recognition (ASR) model, and (b) a vector encoding the speaker information as speaker 
embedding, formed by a concatenation of x-vector and ECAPA-TDNN embeddings. Using a 
previously trained Wasserstein GAN to convert random noise into natural-like yet artificial speaker vectors, we 
randomly sample a new target speaker embedding. If this target vector is dissimilar enough from the original speaker 
embedding (b) - measured by cosine distance -, we use this target speaker and the phone sequence (a) to resynthesize 
the utterance with a custom Speech Synthesis model, resulting in an anonymous version of the original audio. 

In the extension described in [this paper](https://arxiv.org/abs/2210.07002), we mainly use the same pipeline and 
models as in the first version of the system (see the branch [phonetic_representations](https://github.com/DigitalPhonetics/speaker-anonymization/tree/phonetic_representations)). However, the ASR model has been further 
improved by hyperparameter optimization, and the GAN-based anonymization has been introduced as a novel speaker 
selection/generation method.

![architecture](figures/architecture.png)

The current code on the main branch expects the models of release v1.2. Please make sure to download these models 
before running the code.

### Wasserstein GAN
The Generative Adversarial Network (GAN) is trained to minimize the Wasserstein distance between the original and 
generated distribution, hence called Wasserstein GAN, or WGAN. It consists of two parts: a generator that converts 
random noise into a vector of the same shape as our speaker embeddings, and a critic which is responsible for 
decreasing the distance. Since unlike to vanilla GANs, our discriminator (the critic) does not decide between real or 
fake data points but instead compares their distribution, we reduce the chance of common problems like mode collaps and 
the imitation of training data points. To further increase the chance of convergence (another common problem with GANs), 
we train the model with Quadratic Transport Cost. 

During training, we compare the vectors generated by our network to the speaker embeddings of real speakers which 
where extracted on utterance level, meaning that one speaker is represented in this speaker pool as many times as we 
consider utterances from this speaker. In this way, we increase the number of data points in our training data. 
However, GANs are usually trained on larger datasets than the ones considered in the framework of the Voice Privacy 
Challenge, which we follow, and on image data which higher dimensions than speaker embeddings. Therefore, we reduced 
the size of the input noise and the size of the generator and critic ResNet models. More information about this and 
our hyperparameter selection is given in our paper.

During inference, we simply generate a vector using our GAN (or sample one vector of pre-generated vectors) and 
compare it to the speaker embedding of the input speaker. If both vectors have a cosine distance of at least 0.3, we 
consider both vectors (i.e. speakers) as dissimilar enough and take the generated one as target speaker embedding.

If you want to know more about the specifics of this GAN architecture and training, read the original papers about 
[GANs](https://arxiv.org/abs/1406.2661), [Wasserstein GANs](https://proceedings.mlr.press/v70/arjovsky17a.html) and 
their [improvements](https://arxiv.org/abs/1704.00028), and [Wasserstein GANs with Quadratic Transport Cost](https://ieeexplore.ieee.org/document/9009084).


### ASR optimization
We achieved an improvement of our ASR model by applying the following changes:

1. Proportion of Encoder CTC loss during training is 0.6 instead of 0.3, and proportion of Decoder Cross Entropy loss is 0.4 instead of 0.7, meaning more priority for having phone-discriminating representations in the Encoder's output.

2. Gradient accumulation during training is 8 steps instead of 4, meaning larger virtual batch size.

3. Proportion of CTC score during inference is 0.2 instead of 0.4, meaning that slightly less information is used from the input represented by Encoder and slightly more information is used from the language patterns learned by Decoder.

4. Inference is using average of 10 best checkpoints (by Decoder's accuracy on validation data) instead of 1, usually this makes the model less biased towards the validation set resulting in the better generalization on unseen data.

## Installation
### 1. Clone repository
Clone this repository with all its submodules:
```
git clone --recurse-submodules https://github.com/DigitalPhonetics/speaker-anonymization.git
``` 

### 2. Download models
Download the models [from the release page (v1.2)](https://github.com/DigitalPhonetics/speaker-anonymization/releases/tag/v1.2), unzip the folders and place them into a *models* 
folder as stated in the release notes. Make sure to not unzip the single ASR models, only the outer folder.
```
cd speaker-anonymization
mkdir models
cd models
for file in anonymization asr tts; do
    wget https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v1.2/${file}.zip
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
3. TTS: Synthesis based on recognized transcription and anonymized speaker embedding, output as 
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


[comment]: <> (## Citation)

[comment]: <> (```)

[comment]: <> (@inproceedings{meyer_2023_anonymizing,)

[comment]: <> (  author={Sarina Meyer and Pascal Tilli and Pavel Denisov and Florian Lux and Julia Koch and Ngoc Thang Vu},)

[comment]: <> (  title={{Anonymizing Speech with Generative Adversarial Networks to Preserve Speaker Privacy}},)

[comment]: <> (  year=2023,)

[comment]: <> (  booktitle={Proc. SLT 2022},)

[comment]: <> (  pages={},)

[comment]: <> (  doi={})

[comment]: <> (})

[comment]: <> (```)
