#!/bin/bash
# Extract of Voice-Privacy-Challenge-2020/baseline/run.sh
#
# License of the original script:
# Copyright (C) 2020  <Brij Mohan Lal Srivastava, Natalia Tomashenko, Xin Wang, Jose Patino, Paul-Gauthier Noé, Andreas Nautsch, ...>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


set -e

#===== begin config =======

nj=$(nproc)
mcadams=false
stage=0

vp_dir=../Voice-Privacy-Challenge-2020/baseline
cd $vp_dir

download_full=false  # If download_full=true all the data that can be used in the training/development will be dowloaded (except for Voxceleb-1,2 corpus); otherwise - only those subsets that are used in the current baseline (with the pretrained models)
data_url_librispeech=www.openslr.org/resources/12  # Link to download LibriSpeech corpus
data_url_libritts=www.openslr.org/resources/60     # Link to download LibriTTS corpus
corpora=corpora

. utils/parse_options.sh || exit 1;

. path.sh
. cmd.sh

#=========== end config ===========

# Download datasets
if [ $stage -le 0 ]; then
  for dset in libri vctk; do
    for suff in dev test; do
      printf "${GREEN}\nStage 0: Downloading ${dset}_${suff} set...${NC}\n"
      local/download_data.sh ${dset}_${suff} || exit 1;
    done
  done
fi

# Download pretrained models
if [ $stage -le 1 ]; then
  printf "${GREEN}\nStage 1: Downloading pretrained models...${NC}\n"
  local/download_models.sh || exit 1;
fi
data_netcdf=$(realpath exp/am_nsf_data)   # directory where features for voice anonymization will be stored
mkdir -p $data_netcdf || exit 1;

if ! $mcadams; then

  # Download  VoxCeleb-1,2 corpus for training anonymization system models
  if $download_full && [[ $stage -le 2 ]]; then
    printf "${GREEN}\nStage 2: In order to download VoxCeleb-1,2 corpus, please go to: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/ ...${NC}\n"
    sleep 10;
  fi

  # Download LibriSpeech data sets for training anonymization system (train-other-500, train-clean-100)
  if $download_full && [[ $stage -le 3 ]]; then
    printf "${GREEN}\nStage 3: Downloading LibriSpeech data sets for training anonymization system (train-other-500, train-clean-100)...${NC}\n"
    for part in train-clean-100 train-other-500; do
      local/download_and_untar.sh $corpora $data_url_librispeech $part LibriSpeech || exit 1;
    done
  fi

  # Download LibriTTS data sets for training anonymization system (train-clean-100)
  if $download_full && [[ $stage -le 4 ]]; then
    printf "${GREEN}\nStage 4: Downloading LibriTTS data sets for training anonymization system (train-clean-100)...${NC}\n"
    for part in train-clean-100; do
      local/download_and_untar.sh $corpora $data_url_libritts $part LibriTTS || exit 1;
    done
  fi

  # Download LibriTTS data sets for training anonymization system (train-other-500)
  if [ $stage -le 5 ]; then
    printf "${GREEN}\nStage 5: Downloading LibriTTS data sets for training anonymization system (train-other-500)...${NC}\n"
    for part in train-other-500; do
      local/download_and_untar.sh $corpora $data_url_libritts $part LibriTTS || exit 1;
    done
  fi

  libritts_corpus=$(realpath $corpora/LibriTTS)       # Directory for LibriTTS corpus
  librispeech_corpus=$(realpath $corpora/LibriSpeech) # Directory for LibriSpeech corpus

fi # ! $mcadams