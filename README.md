# Multiple-choice-VQA
This project is for the Foundation Models lecture from the University of Stuttgart. The Task is Multiple choice Visual Question Answering

## Table of contents
   1. [Installation](#installation)
   1. [Dataset](#dataset)
   1. [CLIP and T5 Approach](#clip-and-t5-approach)
   1. [BLIP Approach](#blip-approach)
   1. [GradCAM](#gradcam)
      
## Installation
Code has been tested on Python 3.9 and Pytorch 2.1.0, to install all dependencies run :
```
pip install -r requirements.txt
```
## Dataset
Images, questions and MC answers used for training in this work can be found in [VQA v1 Dataset](https://visualqa.org/vqa_v1_download.html). 

## CLIP and T5 Approach
First approach adopts CLIP similarly to zero-shot recognition, employing an MLP to integrate question and image embeddings. Subsequently, it evaluates the output against answer embeddings using cosine similarity to identify the answer with the highest score. Additionally, T5 was explored as a means to enhance this approach, serving as a prompt engineer for CLIP.

<p align="center">
  <img width="400" height="300" src="/media/first_approach.png">
</p>

1. To check demonstrations provided for T5, navigate to [demonstration_t5.json](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/TemplateGeneration_T5/demonstration_t5.json). Afterwards, to evaluate the model's capability as a template generator for answers using these demonstrations, run the notebook located at [t5_approach.ipynb](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/TemplateGeneration_T5/t5_approach.ipynb).
2. To train the CLIP or CLIP+T5 approach, execute the notebook named [Train NN - T5+CLIP.ipynb](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/Train%20NN%20-%20T5%2BCLIP.ipynb). To switch from the clip-only to the T5-enhanced approach, modify the `sentences` variable to `True`. Additionally, you can select the neural network architecture to train from the [models.py](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/models.py) file.<br>For a special architecture experiment using siamese networks instead of the traditional MLP, refer to the notebook titled [Train Siamese NN - T5+CLIP.ipynb](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/Train%20Siamese%20NN%20-%20T5%2BCLIP.ipynb). However, the results were found to be inferior compared to those of a MLP.
3. To precalculate CLIP and CLIP+T5 embeddings for faster training, refer to `compute_store` method from [VQA_Dataset_CLIP](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/VQA_Dataset_CLIP.py#L273) and [VQA_Dataset_CLIP_T5](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/TemplateGeneration_T5/VQA_Dataset_CLIP_T5.py#L27), respectively.
   
## BLIP Approach
This approach uses the BLIP model for the VQA task. To use it for Multiple Choice VQA we us the rank mode. There the model does not produce a new sentence, but ranks the possible answers, in order of the likeliness that the decoder would generate this answer.

<p align="center">
  <img width="500" height="250" src="/media/blip_image.png">
</p>

1. To evaluate and train BLIP, execure notebook named [blip_approach.ipynb](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/BLIP/blip_approach.ipynb)
2. To use the real or abstract dataset use only execute the one corresponding cell
3. To precalculate BLIP preprocessing/tokenization run also the corresponding cell in the same notebook

## GradCAM
<p align="center">
  <img width="900" height="300" src="/media/gradcam_blip.PNG">
</p>

Even though Grad-CAM was originally proposed for CNN approaches, its idea can be extrapolated to the BLIP transformer-based architecture. For this purpose, for BLIP Image-grounded text encoder is chosen and to analyze the significance of attention maps, gradients of the correct Multiple Choice (MC) answer are calculated with respect to the cross-attention maps at a layer determined empirically, notably the 8th layer. This empirical selection process idea performed in [ALBEF](https://github.com/salesforce/ALBEF), aims to identify the layer specializing in certain tasks. We found the 8th BLIP layer was specialized in localization, to apply GradCAM in this layer or any other, please refer to the notebook [GradCam_BLIP.ipynb](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/BLIP/GradCam_BLIP.ipynb).

For Heatmaps for the CLIP model, run notebook [CLIP_GradCAM_Visualization.ipynb](https://github.com/JansSolanoVega/Multiple-choice-VQA/blob/main/CLIP_GradCAM_Visualization.ipynb). Which is a modified version from
[A Playground for CLIP-like Models](https://github.com/kevinzakka/clip_playground/tree/main) repository.
