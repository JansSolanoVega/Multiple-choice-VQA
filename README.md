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

## BLIP Approach

## Grad-CAM

## References
