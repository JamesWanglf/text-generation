# text-generation
### Table of Contents  
[1. Summary](#summary)  
[2. Prerequisite](#prerequisite)  
[3. Installation](#installation)  

## <a name="summary"/>1. Summary
  - ### Image captioning
    This generates the textual description of an image. This is encoder-decoder architecture.  
    The Convolutional Neural Network(CNN) can be thought of as an encoder. The input image is given to CNN to extract the features. The last hidden state of the CNN is connected to the Decoder.  
    The Decoder is a Recurrent Neural Network(RNN) which does language modelling up to the word level.
  - ### Text adventure game

## <a name="prerequisite"/>2. Prerequisite
1. Install Nvidia drivers on ubuntu-18.04 machine.
2. Install CUDA toolkit 10.2.
3. Install cudnn8.2.1.


## <a name="installation"/>3. Installation
1. ```conda create -n text-generation-v1 python=3.7```
2. ```conda activate text-generation-v1```
3. ```pip install -r requirements.txt```
