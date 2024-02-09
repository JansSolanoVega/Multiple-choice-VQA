# Multiple-choice-VQA

## Table of contents
   1. [Installation](#installation)
   1. [Dataset](#dataset)
   1. [CLIP+T5 Approach](#clip+t5)
   1. [BLIP Approach](#tutorials)
   1. [Grad-CAM](#gradcam)
   1. [References](#references-and-citation)

## Installation
Code has been tested on Python 3.9 and Pytorch 2.1.0, to install all dependencies run :
```
pip install -r requirements.txt
```
## Dataset
Images, questions and MC answers used for training in this work can be found in [VQA v1 Dataset](https://visualqa.org/vqa_v1_download.html). 

## CLIP+T5 Approach
First approach adopts CLIP similarly to zero-shot recognition, employing an MLP to integrate question and image embeddings. Subsequently, it evaluates the output against answer embeddings using cosine similarity to identify the answer with the highest score. Additionally, T5 was explored as a means to enhance this approach, serving as a prompt engineer for CLIP.

1. To check demonstrations provided for T5, navigate to `TemplateGeneration_T5/demonstration_t5.json`. Afterwards, to evaluate the model's capability as a template generator for answers using these demonstrations, run the notebook located at `TemplateGeneration_T5/t5_approach.ipynb`.
2. To train the CLIP or CLIP+T5 approach, execute the notebook named `Train NN - T5+CLIP.ipynb`. To switch from the clip-only to the T5-enhanced approach, modify the `sentences` variable to `True`. Additionally, you can select the neural network architecture to train from the `models.py` folder.

   For a special architecture experiment using siamese networks instead of the traditional MLP, refer to the notebook titled `Train Siamese NN - T5+CLIP.ipynb`. However, the results were found to be inferior compared to those of a MLP.
 
## BLIP Approach

## Grad-CAM

## References
