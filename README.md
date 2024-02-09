# Multiple-choice-VQA

## Table of contents
   1. [Installation](#installation)
   1. [Dataset](#dataset)
   1. [CLIP+T5 Approach](CLIP+T5-Approach)
   1. [BLIP Approach](#BLIP-Approach)
   1. [Grad-CAM](#Grad-CAM)
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
2. To train the CLIP or CLIP+T5 approach, execute the notebook named `Train NN - T5+CLIP.ipynb`. To switch from the clip-only to the T5-enhanced approach, modify the `sentences` variable to `True`. Additionally, you can select the neural network architecture to train from the `models.py` folder.<br>For a special architecture experiment using siamese networks instead of the traditional MLP, refer to the notebook titled `Train Siamese NN - T5+CLIP.ipynb`. However, the results were found to be inferior compared to those of a MLP.
3. To precalculate CLIP and CLIP+T5 embeddings for faster training, refer to `compute_store` method from [VQA_Dataset_CLIP](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/VQA_Dataset_CLIP.py#L273) and [VQA_Dataset_CLIP_T5](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/TemplateGeneration_T5/VQA_Dataset_CLIP_T5.py#L27), respectively. 
 
## BLIP Approach

## Grad-CAM
Even though Grad-CAM was originally proposed for CNN approaches, its idea can be extrapolated to the BLIP transformer-based architecture. For this purpose, for BLIP Image-grounded text encoder is chosen and to analyze the significance of attention maps, gradients of the correct Multiple Choice (MC) answer are calculated with respect to the cross-attention maps at a layer determined empirically, notably the 8th layer. This empirical selection process idea performed in [ALBEF](https://github.com/salesforce/ALBEF), aims to identify the layer specializing in certain tasks, such as, in our case, localizations. For getting BLIP heatmaps with Grad-CAM, run the notebook [GradCam_BLIP.ipynb](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/BLIP/GradCam_BLIP.ipynb).
## References
