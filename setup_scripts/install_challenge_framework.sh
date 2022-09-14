#!/bin/bash

# Modified copy of the install script of Voice-Privacy-Challenge-2020

set -e

nj=$(nproc)

cd ../Voice-Privacy-Challenge-2020
home=$PWD

conda_url=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
venv_dir=$PWD/../venv

cuda_path="Path/to/cuda-11.6"  # TODO: change path to cuda
mkl_root="Path/to/mkl"  # TODO: change path to mkl

mark=.done-venv
if [ ! -f $mark ]; then
  echo 'Making python virtual environment'
  name=$(basename $conda_url)
  if [ ! -f $name ]; then
    wget $conda_url || exit 1
  fi
  [ ! -f $name ] && echo "File $name does not exist" && exit 1
  [ -d $venv_dir ] && rm -r $venv_dir
  sh $name -b -p $venv_dir || exit 1
  . $venv_dir/bin/activate
  echo 'Installing python dependencies'
  pip install -r requirements.txt || exit 1
  touch $mark
fi
echo "if [ \$(which python) != $venv_dir/bin/python ]; then source $venv_dir/bin/activate; fi" > env.sh
export PATH=${cuda_path}/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${cuda_path}/lib64
export CUDA_HOME=${cuda_path}
echo "export PATH=${cuda_path}/bin:\$PATH" >> env.sh
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${cuda_path}/lib64" >> env.sh
echo "export CUDA_HOME=${cuda_path}" >> env.sh


mark=.done-kaldi-tools
if [ ! -f $mark ]; then
  echo 'Building Kaldi tools'
  cd kaldi/tools
  extras/check_dependencies.sh || exit 1
  make -j $nj || exit 1
  cd $home
  touch $mark
fi

mark=.done-kaldi-src
if [ ! -f $mark ]; then
  echo 'Building Kaldi src'
  cd kaldi/src
  ./configure --shared --mkl-root=${mkl_root} --cudatk-dir=${cuda_path} --with-cudadecoder=no || exit 1
  make clean || exit 1
  make depend -j $nj || exit 1
  make -j $nj || exit 1
  cd $home
  touch $mark
fi

echo Done
