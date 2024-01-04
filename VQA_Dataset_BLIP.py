import datasets
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
from PIL import Image
from torchvision import transforms
import os
import h5py
import matplotlib.pyplot as plt

import numpy as np
from torchvision.transforms.functional import InterpolationMode

class VQA_Dataset(torch.utils.data.Dataset):
    
    def __init__(self):
        self.images = []
        self.questions = []
        self.answers = []
        self.correct_answers = [] 
        
    def load_all(self, preprocess=None, device="cuda", length=100):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(script_dir,'Annotations/MultipleChoice_abstract_v002_val2015_questions.json'), 'r') as question_file:
            with open(os.path.join(script_dir,'Annotations/abstract_v002_val2015_annotations.json'), 'r') as answer_file:
                question_data = json.load(question_file)
                answer_data = json.load(answer_file)

                annntoations = (answer_data['annotations'])
                questions = (question_data['questions'])

                for question in tqdm(questions[:length], desc="Preprocessing Images"):
                    
                    image_id = question['image_id']
                    question_text = question['question']
                    question_id = question['question_id']
                    answers_text = question['multiple_choices']

                    # find quesion id and image id in annotations
                    for annotation in annntoations:
                        if annotation['question_id'] == question_id and annotation['image_id'] == image_id:
                            correct_answer = annotation['multiple_choice_answer']
                            # get the index of the answer in the list of answers
                            for i, possible_answer in enumerate(answers_text):
                                if possible_answer == correct_answer:
                                    index = i
                                    break

                    self.correct_answers.append(answers_text[index])
                    self.questions.append(question_text)
                    self.answers.append(answers_text)

                    img = Image.open(os.path.join(script_dir,"Images/abstract_v002_val2015_0000000{}.png".format(image_id)))
                    self.images.append(img)
                    


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        return  self.images[index], self.questions[index], self.answers[index], self.correct_answers[index]

class VQA_Dataset_TorchVersion(torch.utils.data.Dataset):
    
    def __init__(self):
        self.images = []
        self.questions = []
        self.answers = []
        self.correct_answers = [] 

        self.image_size = 480
        
    def load_all(self, device="cuda", length=100):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(script_dir,'Annotations/MultipleChoice_abstract_v002_val2015_questions.json'), 'r') as question_file:
            with open(os.path.join(script_dir,'Annotations/abstract_v002_val2015_annotations.json'), 'r') as answer_file:
                question_data = json.load(question_file)
                answer_data = json.load(answer_file)

                annntoations = (answer_data['annotations'])
                questions = (question_data['questions'])

                for question in tqdm(questions[:length], desc="Preprocessing Images"):
                    
                    image_id = question['image_id']
                    question_text = question['question']
                    question_id = question['question_id']
                    answers_text = question['multiple_choices']

                    # find quesion id and image id in annotations
                    for annotation in annntoations:
                        if annotation['question_id'] == question_id and annotation['image_id'] == image_id:
                            correct_answer = annotation['multiple_choice_answer']
                            # get the index of the answer in the list of answers
                            for i, possible_answer in enumerate(answers_text):
                                if possible_answer == correct_answer:
                                    index = i
                                    break

                    self.correct_answers.append(answers_text[index])
                    self.questions.append(question_text)
                    self.answers.append(answers_text)

                    raw_image = Image.open(os.path.join(script_dir,"Images/abstract_v002_val2015_0000000{}.png".format(image_id))).convert('RGB')
                    transform = transforms.Compose([
                        transforms.Resize((self.image_size,self.image_size),interpolation=InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                        ]) 
                    img = transform(raw_image).unsqueeze(0).to(device)
                    self.images.append(img)
                    


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        return  self.images[index], self.questions[index], self.answers[index], self.correct_answers[index]

class VQA_Dataset_preloaded_TorchVersion(torch.utils.data.Dataset):
    
    def __init__(self, device, image_size=480, folder_path="H:/FoundationModels/"):
        self.file = None
        self.imgs = None
        self.questions = None
        self.multiple_answers = None
        self.correct_answers = None
        self.path = folder_path
        self.max_length = 20
        self.image_height = image_size
        self.image_width = image_size
        self.num_answers = 18
        self.device = device
        
    def compute_store(self, tokenizer, length=100, fileName="embeddingsBLIP.h5", name="train", version="real"):
        """
        preprocess images and tokenize questions and answers and compute embeddings
        store them in a file
        """
        self.fileName = fileName
        self.tokenizer = tokenizer

        script_dir = os.path.dirname(os.path.abspath(__file__))
        if version == "real":
            path_questions = f"Annotations/MultipleChoice_mscoco_{name}2014_questions.json"
            path_questions = os.path.join(script_dir, path_questions)
            path_answers = f"Annotations/mscoco_{name}2014_annotations.json"
            path_answers = os.path.join(script_dir, path_answers)
        else:
            # abstract
            # only val available
            path_questions = os.path.join(script_dir,'Annotations/MultipleChoice_abstract_v002_val2015_questions.json')
            path_answers = os.path.join(script_dir,'Annotations/abstract_v002_val2015_annotations.json')
        with open(path_questions, 'r') as question_file:
            with open(path_answers, 'r') as answer_file:
                question_data = json.load(question_file)
                answer_data = json.load(answer_file)

                annntoations = (answer_data['annotations'])
                questions = (question_data['questions'])
                
                with h5py.File(self.path + self.fileName, 'a') as hf:
                    for idx, question in enumerate(tqdm(questions[:length], desc="Preprocessing Images")):
                        if idx == 0:
                            imgs_ds = hf.create_dataset('imgs', shape=(length, 3, self.image_height, self.image_width))#, compression="gzip")#, maxshape=(length, 3, self.image_height, self.image_width))
                            questions_input_ids_ds = hf.create_dataset('questions_input_ids', shape=(length, self.max_length))#, compression="gzip")#, maxshape=(length, self.max_length))
                            questions_attention_mask_ds = hf.create_dataset('questions_attention_mask', shape=(length, self.max_length))#, compression="gzip")#, maxshape=(length, self.max_length))
                            multiple_answers_input_ids_ds = hf.create_dataset('multiple_answers_input_ids', shape=(length, self.num_answers, self.max_length))#, compression="gzip")#, maxshape=(length, self.num_answers, None))
                            multiple_answers_attention_mask_ds = hf.create_dataset('multiple_answers_attention_mask', shape=(length, self.num_answers, self.max_length))#, compression="gzip")#, maxshape=(length, self.num_answers, None))
                            correct_answers_input_ids_ds = hf.create_dataset('correct_answers_input_ids', shape=(length, self.max_length))#, compression="gzip")#, maxshape=(length, self.max_length))
                            correct_answers_attention_mask_ds = hf.create_dataset('correct_answers_attention_mask', shape=(length, self.max_length))#, compression="gzip")#, maxshape=(length, self.max_length))

                            
                        image_id = question['image_id']
                        question_id = question['question_id']
                        answers_text = question['multiple_choices']
                        
                        # find quesion id and image id in annotations
                        for annotation in annntoations:
                            if annotation['question_id'] == question_id and annotation['image_id'] == image_id:
                                correct_answer = annotation['multiple_choice_answer']
                                # get the index of the answer in the list of answers
                                for i, possible_answer in enumerate(answers_text):
                                    if possible_answer == correct_answer:
                                        index = i
                                        break

                        #Image encoding 
                        if version == "real":
                            image_id = str(image_id).zfill(12)
                            raw_image = Image.open("Images_real/" + name + "2014/COCO_" + name + "2014_{}.jpg".format(image_id)).convert('RGB')
                        else:

                            raw_image = Image.open(os.path.join(script_dir,"Images/abstract_v002_val2015_0000000{}.png".format(image_id))).convert('RGB')
                        transform = transforms.Compose([
                            transforms.Resize((self.image_height,self.image_width),interpolation=InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                            ])
                        img = transform(raw_image).unsqueeze(0)

                        ##Question encoding
                        question_text = question['question']
                        question = self.tokenizer(question_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device) 
                        question.input_ids[:,0] = self.tokenizer.enc_token_id

                        ##Correct answer encoding
                        correct_answer_text = answers_text[index]   
                        correct_answer = self.tokenizer(correct_answer_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device) 
                        correct_answer.input_ids[:,0] = self.tokenizer.enc_token_id

                        ##MC answers encoding
                        multiple_answers = self.tokenizer(answers_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                        multiple_answers.input_ids[:,0] = self.tokenizer.bos_token_id

                        #Store with h5py
                        imgs_ds[idx] = img.cpu().numpy()

                        correct_answers_input_ids_ds[idx] = correct_answer.input_ids.detach().cpu().numpy()[0]
                        correct_answers_attention_mask_ds[idx] = correct_answer.attention_mask.detach().cpu().numpy()[0]

                        questions_input_ids_ds[idx] = question.input_ids.detach().cpu().numpy()[0]
                        questions_attention_mask_ds[idx] = question.attention_mask.detach().cpu().numpy()[0]

                        multiple_answers_input_ids_ds[idx] = multiple_answers.input_ids.detach().cpu().numpy()
                        multiple_answers_attention_mask_ds[idx] = multiple_answers.attention_mask.detach().cpu().numpy()


    def compute_store_sentences(self, tokenizer, length=100, fileName="embeddingsBLIP.h5", name="train", version="real"):
        """
        preprocess images and tokenize questions and answers and compute embeddings
        store them in a file
        """
        self.fileName = fileName
        self.tokenizer = tokenizer

        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir,'sentences_' + name + '.json'), 'r') as file:
                
                data = json.load(file)
                data = data['data']
                
                with h5py.File(self.path + self.fileName, 'a') as hf:
                    for idx, pair in enumerate(tqdm(data[:length], desc="Preprocessing Images")):
                        if idx == 0:
                            imgs_ds = hf.create_dataset('imgs', shape=(length, 3, self.image_height, self.image_width))#, maxshape=(length, 3, self.image_height, self.image_width))
                            questions_input_ids_ds = hf.create_dataset('questions_input_ids', shape=(length, self.max_length))#, maxshape=(length, self.max_length))
                            questions_attention_mask_ds = hf.create_dataset('questions_attention_mask', shape=(length, self.max_length))#, maxshape=(length, self.max_length))
                            multiple_answers_input_ids_ds = hf.create_dataset('multiple_answers_input_ids', shape=(length, self.num_answers, self.max_length))#, maxshape=(length, self.num_answers, None))
                            multiple_answers_attention_mask_ds = hf.create_dataset('multiple_answers_attention_mask', shape=(length, self.num_answers, self.max_length))#, maxshape=(length, self.num_answers, None))
                            correct_answers_input_ids_ds = hf.create_dataset('correct_answers_input_ids', shape=(length, self.max_length))#, maxshape=(length, self.max_length))
                            correct_answers_attention_mask_ds = hf.create_dataset('correct_answers_attention_mask', shape=(length, self.max_length))#, maxshape=(length, self.max_length))

                        
                        image_id = pair['image_id']
                        question_id = pair['question_id']
                        answers_text = pair['answers']
                            
                        
                        # find quesion id and image id in annotations
                        
                        correct_answer = pair['correct_answer']
                        #Image encoding 
                        if version == "real":
                            image_id = str(image_id).zfill(12)
                            raw_image = Image.open("Images_real/" + name + "2014/COCO_" + name + "2014_{}.jpg".format(image_id)).convert('RGB')
                        else:

                            raw_image = Image.open(os.path.join(script_dir,"Images/abstract_v002_val2015_0000000{}.png".format(image_id))).convert('RGB')
                        transform = transforms.Compose([
                            transforms.Resize((self.image_height,self.image_width),interpolation=InterpolationMode.BICUBIC),
                            transforms.ToTensor()
                            #,transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                            ])
                        img = transform(raw_image).unsqueeze(0)

                        ##Question encoding
                        
                        question_text = pair['question']
                        question = self.tokenizer(question_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device) 
                        question.input_ids[:,0] = self.tokenizer.enc_token_id

                        ##Correct answer encoding
                        correct_answer_text = answers_text[correct_answer]   
                        correct_answer = self.tokenizer(correct_answer_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device) 
                        correct_answer.input_ids[:,0] = self.tokenizer.enc_token_id

                        ##MC answers encoding
                        multiple_answers = self.tokenizer(answers_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                        multiple_answers.input_ids[:,0] = self.tokenizer.bos_token_id

                        #Store with h5py
                        imgs_ds[idx] = img.cpu().numpy()

                        correct_answers_input_ids_ds[idx] = correct_answer.input_ids.detach().cpu().numpy()[0]
                        correct_answers_attention_mask_ds[idx] = correct_answer.attention_mask.detach().cpu().numpy()[0]

                        questions_input_ids_ds[idx] = question.input_ids.detach().cpu().numpy()[0]
                        questions_attention_mask_ds[idx] = question.attention_mask.detach().cpu().numpy()[0]

                        multiple_answers_input_ids_ds[idx] = multiple_answers.input_ids.detach().cpu().numpy()
                        multiple_answers_attention_mask_ds[idx] = multiple_answers.attention_mask.detach().cpu().numpy()
    


    def load(self, fileName="embeddingsBLIP.h5", length=-1):
        self.fileName = fileName
        self.file = h5py.File(self.path + fileName, 'r')
        self.imgs = self.file["imgs"]#[:length]
        
        self.length = self.imgs.shape[0]
        self.file.close()
        """
        self.questions_input_ids = self.file["questions_input_ids"]#[:length]
        self.questions_attention_mask = self.file["questions_attention_mask"]#[:length]
        self.multiple_answers_input_ids = self.file["multiple_answers_input_ids"]#[:length]
        self.multiple_answers_attention_mask = self.file["multiple_answers_attention_mask"]#[:length]
        self.correct_answers_input_ids = self.file["correct_answers_input_ids"]#[:length]
        self.correct_answers_attention_mask = self.file["correct_answers_attention_mask"]#[:length]
        """


    def __len__(self):
        return self.length


    def __getitem__(self, index):
        #if index % 100 == 0:  # Close and reopen the file every 100 batches
        #    print("getitem_100")
        #    self.file.close()
        
        with h5py.File(self.path + self.fileName, 'r') as hf:
            imgs = torch.from_numpy(hf["imgs"][index]).unsqueeze(0).to(self.device)
            question = {"input_ids": torch.from_numpy(hf["questions_input_ids"][index]).type(torch.int64).unsqueeze(0).to(self.device), "attention_mask":torch.from_numpy(hf["questions_attention_mask"][index]).type(torch.int64).unsqueeze(0).to(self.device)}
            multiple_answer = {"input_ids": torch.from_numpy(hf["multiple_answers_input_ids"][index]).type(torch.int64).to(self.device), "attention_mask":torch.from_numpy(hf["multiple_answers_attention_mask"][index]).type(torch.int64).to(self.device)}
            correct_answer = {"input_ids": torch.from_numpy(hf["correct_answers_input_ids"][index]).type(torch.int64).unsqueeze(0).to(self.device), "attention_mask":torch.from_numpy(hf["correct_answers_attention_mask"][index]).type(torch.int64).unsqueeze(0).to(self.device)}
            encoding = {'imgs':imgs, 'questions': question, 'multiple_answers': multiple_answer, 'correct_answers': correct_answer}
        
        encoding = {'imgs':imgs, 'questions': question, 'multiple_answers': multiple_answer, 'correct_answers': correct_answer}
        
        # delete variables to save memory
        
        #del question, multiple_answer, correct_answer, imgs

        """
        question = {"input_ids": torch.from_numpy(self.questions_input_ids[index]).type(torch.int64).unsqueeze(0).to(self.device), "attention_mask":torch.from_numpy(self.questions_attention_mask[index]).type(torch.int64).unsqueeze(0).to(self.device)}
        multiple_answer = {"input_ids": torch.from_numpy(self.multiple_answers_input_ids[index]).type(torch.int64).to(self.device), "attention_mask":torch.from_numpy(self.multiple_answers_attention_mask[index]).type(torch.int64).to(self.device)}
        correct_answer = {"input_ids": torch.from_numpy(self.correct_answers_input_ids[index]).type(torch.int64).unsqueeze(0).to(self.device), "attention_mask":torch.from_numpy(self.correct_answers_attention_mask[index]).type(torch.int64).unsqueeze(0).to(self.device)}
        imgs = torch.from_numpy(self.imgs[index]).unsqueeze(0).to(self.device)
        encoding = {'imgs':imgs, 'questions': question, 'multiple_answers': multiple_answer, 'correct_answers': correct_answer}
        """
        return encoding

class VQA_Dataset_preloaded_alternativ(torch.utils.data.Dataset):
    
    def __init__(self, device, image_size=480, folder_path="H:/FoundationModels/"):
        self.file = None
        self.imgs = None
        self.questions = None
        self.multiple_answers = None
        self.correct_answers = None
        self.path = folder_path
        self.max_length = 20
        self.image_height = image_size
        self.image_width = image_size
        self.num_answers = 18
        self.device = device
      
    def load(self, fileName="embeddingsBLIP.h5", length=-1):
        self.fileName = fileName
        print(self.path + fileName)
        self.file = h5py.File(self.path + fileName, 'r')
        self.imgs = self.file["imgs"]#[:length]
        
        self.length = self.imgs.shape[0]
        self.questions_input_ids = self.file["questions_input_ids"]#[:length]
        self.questions_attention_mask = self.file["questions_attention_mask"]#[:length]
        self.multiple_answers_input_ids = self.file["multiple_answers_input_ids"]#[:length]
        self.multiple_answers_attention_mask = self.file["multiple_answers_attention_mask"]#[:length]
        self.correct_answers_input_ids = self.file["correct_answers_input_ids"]#[:length]
        self.correct_answers_attention_mask = self.file["correct_answers_attention_mask"]#[:length]
        


    def __len__(self):
        return self.length


    def __getitem__(self, index):
        #if index % 100 == 0:  # Close and reopen the file every 100 batches
        #    print("getitem_100")
        #    self.file.close()
        """
        with h5py.File(self.path + self.fileName, 'r') as hf:
            imgs = torch.from_numpy(hf["imgs"][index]).unsqueeze(0).to(self.device)
            question = {"input_ids": torch.from_numpy(hf["questions_input_ids"][index]).type(torch.int64).unsqueeze(0).to(self.device), "attention_mask":torch.from_numpy(hf["questions_attention_mask"][index]).type(torch.int64).unsqueeze(0).to(self.device)}
            multiple_answer = {"input_ids": torch.from_numpy(hf["multiple_answers_input_ids"][index]).type(torch.int64).to(self.device), "attention_mask":torch.from_numpy(hf["multiple_answers_attention_mask"][index]).type(torch.int64).to(self.device)}
            correct_answer = {"input_ids": torch.from_numpy(hf["correct_answers_input_ids"][index]).type(torch.int64).unsqueeze(0).to(self.device), "attention_mask":torch.from_numpy(hf["correct_answers_attention_mask"][index]).type(torch.int64).unsqueeze(0).to(self.device)}
            encoding = {'imgs':imgs, 'questions': question, 'multiple_answers': multiple_answer, 'correct_answers': correct_answer}
        
        encoding = {'imgs':imgs, 'questions': question, 'multiple_answers': multiple_answer, 'correct_answers': correct_answer}
        
        # delete variables to save memory
        
        #del question, multiple_answer, correct_answer, imgs

        return encoding
        """
        # load without dict
        imgs = torch.from_numpy(self.imgs[index]).to(self.device)
        question_input_ids = torch.from_numpy(self.questions_input_ids[index]).type(torch.int64).to(self.device)
        question_attention_mask = torch.from_numpy(self.questions_attention_mask[index]).type(torch.int64).to(self.device)
        multiple_answers_input_ids = torch.from_numpy(self.multiple_answers_input_ids[index]).type(torch.int64).to(self.device)
        multiple_answers_attention_mask = torch.from_numpy(self.multiple_answers_attention_mask[index]).type(torch.int64).to(self.device)
        correct_answers_input_ids = torch.from_numpy(self.correct_answers_input_ids[index]).type(torch.int64).to(self.device)
        correct_answers_attention_mask = torch.from_numpy(self.correct_answers_attention_mask[index]).type(torch.int64).to(self.device)
        
        return imgs, question_input_ids, question_attention_mask, multiple_answers_input_ids, multiple_answers_attention_mask, correct_answers_input_ids, correct_answers_attention_mask
           


class VQA_Dataset_preloaded(torch.utils.data.Dataset):
    
    def __init__(self, device):
        self.file = None
        self.fileName = None

        self.input_ids = None
        self.attention_mask = None
        self.pixel_values = None
        self.labels = None
        self.answer_choices = None

        self.max_length = 11
        self.image_height = 384
        self.image_width = 384
        self.num_answers = 18
        self.device = device
        
    def compute_store(self, text_processor, image_processor, length=100, fileName="embeddingsBLIP.h5"):
        """
        preprocess images and tokenize questions and answers and compute embeddings
        store them in a file
        """
        self.fileName = fileName
        self.text_processor = text_processor
        self.image_processor = image_processor

        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(script_dir,'Annotations/MultipleChoice_abstract_v002_val2015_questions.json'), 'r') as question_file:
            with open(os.path.join(script_dir,'Annotations/abstract_v002_val2015_annotations.json'), 'r') as answer_file:
                question_data = json.load(question_file)
                answer_data = json.load(answer_file)

                annntoations = (answer_data['annotations'])
                questions = (question_data['questions'])
                
                with h5py.File(self.fileName, 'a') as hf:
                    for idx, question in enumerate(tqdm(questions[:length], desc="Preprocessing Images")):
                        if idx == 0:
                            input_ids_ds = hf.create_dataset('input_ids', shape=(length, self.max_length), maxshape=(length, self.max_length))
                            attention_mask_ds = hf.create_dataset('attention_mask', shape=(length, self.max_length), maxshape=(length, self.max_length))
                            pixel_values_ds = hf.create_dataset('pixel_values', shape=(length, 3, self.image_height, self.image_width), maxshape=(length, 3, self.image_height, self.image_width))
                            labels_ds = hf.create_dataset('labels', shape=(length, self.max_length), maxshape=(length, self.max_length))
                            answer_choices_ds = hf.create_dataset('answer_choices', shape=(length, self.num_answers, self.max_length), maxshape=(length, self.num_answers, self.max_length))

                        image_id = question['image_id']
                        question_id = question['question_id']
                        answers_text = question['multiple_choices']
                        
                        # find quesion id and image id in annotations
                        for annotation in annntoations:
                            if annotation['question_id'] == question_id and annotation['image_id'] == image_id:
                                correct_answer = annotation['multiple_choice_answer']
                                # get the index of the answer in the list of answers
                                for i, possible_answer in enumerate(answers_text):
                                    if possible_answer == correct_answer:
                                        index = i
                                        break
                            
                        correct_answer = answers_text[index]                        
                        image = Image.open(os.path.join(script_dir,"Images/abstract_v002_val2015_0000000{}.png".format(image_id))).convert('RGB')
                        question_text = question['question']
                        
                        image_encoding = self.image_processor(image,
                                                            do_resize=True,
                                                            size=(self.image_height,self.image_width),
                                                            return_tensors="pt")

                        encoding = self.text_processor(
                                                None,
                                                question_text,
                                                padding="max_length",
                                                truncation=True,
                                                max_length = self.max_length,
                                                return_tensors="pt"
                                                )
                        # remove batch dimension
                        for k,v in encoding.items():
                            encoding[k] = v.squeeze()
                        encoding["pixel_values"] = image_encoding["pixel_values"][0]
                        
                        # add labels
                        labels = self.text_processor.tokenizer.encode(
                            correct_answer,
                            max_length= self.max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors='pt'
                        )[0]
                        encoding["labels"] = labels

                        # Multiple choices
                        multiple_choices = []
                        for possible_answer in answers_text:
                            multiple_choices.append(self.text_processor.tokenizer.encode(
                                possible_answer,
                                max_length= self.max_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors='pt'
                            )[0])

                        # store with h5py
                        input_ids_ds[idx] = encoding["input_ids"].detach().cpu().numpy()
                        attention_mask_ds[idx] = encoding["attention_mask"].detach().cpu().numpy()
                        pixel_values_ds[idx] = encoding["pixel_values"].detach().cpu().numpy()
                        labels_ds[idx] = encoding["labels"].detach().cpu().numpy()
                        answer_choices_ds[idx] = torch.stack(multiple_choices).detach().cpu().numpy()

    def load(self, fileName="embeddingsBLIP.h5"):
        self.file = h5py.File(fileName, 'r')
        self.input_ids = self.file["input_ids"]
        self.attention_mask = self.file["attention_mask"]
        self.pixel_values = self.file["pixel_values"]
        self.labels = self.file["labels"]  
        self.answer_choices = self.file["answer_choices"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        input_ids = torch.from_numpy(self.input_ids[index]).type(torch.int64).to(self.device)
        attention_mask = torch.from_numpy(self.attention_mask[index]).type(torch.int64).to(self.device)
        pixel_values = torch.from_numpy(self.pixel_values[index]).type(torch.float32).to(self.device)
        labels = torch.from_numpy(self.labels[index]).type(torch.int64).to(self.device)
        answer_choices = torch.from_numpy(self.answer_choices[index]).type(torch.int64).to(self.device)
        encoding = {'input_ids':input_ids, 'attention_mask':attention_mask, 'pixel_values': pixel_values, 'labels':labels, 'answer_choices':answer_choices}
        return  encoding


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
   # model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = VQA_Dataset()
    #dataset.load_all(preprocess, length=200, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i in tqdm(dataloader, desc="Testing"):
        id = i['image_id']
        if id == torch.tensor(26216):
            print("id: ", id, "question: ", i['question'], "Answer: ", i['correct_answer_text'])
            print("Possible answers: ", i['possible_answers'])
            break

