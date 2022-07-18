# text-generation
### Table of Contents  
[1. Summary](#summary)  
[2. Prerequisite](#prerequisite)  
[3. Installation](#installation)  

## <a name="summary"/>1. Summary
  - ### Image captioning
    This module is based on [BLIP](https://github.com/salesforce/BLIP).  
    This module generates the textual description of an image. This is encoder-decoder architecture.  
    The Convolutional Neural Network(CNN) can be thought of as an encoder. The input image is given to CNN to extract the features. The last hidden state of the CNN is connected to the Decoder.  
    The Decoder is a Recurrent Neural Network(RNN) which does language modelling up to the word level.

## <a name="prerequisite"/>2. Prerequisite
1. Install Nvidia drivers on ubuntu-18.04 machine.
2. Install CUDA toolkit 10.2.
3. Install cudnn8.2.1.


## <a name="installation"/>3. Installation
1. Clone this repository.
    ```
    git clone https://github.com/JamesWanglf/text-generation.git
    cd text-generation
    ```
2. ```conda create -n text-generation-v1 python=3.7```
3. ```conda activate text-generation-v1```
4. ```pip install -r requirements.txt```
5. ```pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html```
6. Download the checkpoints  
You can download the pretrained model from [this link](https://drive.google.com/file/d/1JMmqsL7Nrq4B2WUXt6it3-c-4LnVOthz/view?usp=sharing) by running the following command.  
    ```
    cd image_captioning
    mkdir checkpoints
    cd checkpoints
    gdown https://drive.google.com/uc?id=1JMmqsL7Nrq4B2WUXt6it3-c-4LnVOthz -O model_base.pth
    ```
