# Alternative Installation of the Framework for the Voice Privacy Challenge 2020
Unfortunately, the installation is not always as easy as the organizers imply in their [install 
script](Voice-Privacy-Challenge-2020/install.sh), and installs several tools that are only necessary if the primary 
baseline of the challenge should be executed. To adapt the script to our devices and pipeline, we shortened and 
modified it, and exchanged some components.

**Note: To run the code in this repository, it is NOT necessary to use the installation steps described in this 
document. Instead, you can also simply use the original [install 
script](Voice-Privacy-Challenge-2020/install.sh). If you use this document, be aware that you probably have to 
modify several steps to make it work for you.**

## Installation Steps
This guide expects that you cloned the repository included submodules. Once you followed the installation steps 
described in the following, continue with the *Getting started* section in the [main README](README.md).

### 1. Environment creation
The original installation script would create a conda environment but conda would include many packages that are not 
always needed. We therefore 'manually' create a virtual environment within the 
repository:
```
virtualenv venv --python=python3.8
source venv/bin/activate
pip install -r Voice-Privacy-Challenge-2020/requirements.txt
```
Instead of the last line, if you want to install all requirements for the whole repository, you can instead run
```
pip install -r requirements.txt
```
(If this does not work, install the requirements files listed in it separately)

Finally, we have to make the install script skip the step of creating an environment by creating the required check 
file:
```
touch Voice-Privacy-Challenge-2020/.done-venv
```

### 2. Adapting Kaldi
The version of Kaldi in the framework is not up to date, and even the up to date one does not officially support our 
gcc version. We have to change that:
```
cd Voice-Privacy-Challenge-2020/kaldi
git checkout master
vim src/configure
```
In src/configure, change the min supported gcc version:
```
  -          MIN_UNSUPPORTED_GCC_VER="10.0"
  -          MIN_UNSUPPORTED_GCC_VER_NUM=100000;
  +          MIN_UNSUPPORTED_GCC_VER="12.0"
  +          MIN_UNSUPPORTED_GCC_VER_NUM=120000;
```

### 3. CUDA and MKL
Due to several installed versions of CUDA and MKL, and very specific requirements of Kaldi, we have to specify the 
paths to them in the [setup_scripts/install_challenge_framework.sh](../speaker-anonymization/setup_scripts/install_challenge_framework.sh) file. 

### 4. Installation
Once everything above is resolved, you simply have to run the adapted install script:
```
cd setup_scripts
./install_challenge_framework.sh
```
