import torch
import h5py
import json
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
from utils import *
from PIL import Image
import clip

class VQA_Dataset_preloaded(torch.utils.data.Dataset):
    """
    Torch dataset to compute a dataset, where multiple choice answers are already transformed using T5.
    All dataset items are computed as CLIP embeddings for faster training
    """
    def __init__(self):
        self.device = None
        self.file = None
        self.image_features = None
        self.question_features = None
        self.answer_features = None
        self.correct_answers = None
        self.answer_types = None


        
    def compute_store(self, preprocess, model, device, path, name="train",length=100, mode="scale", real=True, sentences=True):
        """
        preprocess images and tokenize questions and answers and compute embeddings
        store them in a file
        """
        self.device = device
        conversion_answer_type = {"yes/no": 0, "number": 1, "other": 2}

        if real:
            imagesType = "mscoco"
        else:
            imagesType = "abstract"
        
        self.fileName = imagesType + "_" + path

        year = {"mscoco": "2014", "abstract": "2015"}
        imgname_start = {"mscoco": "COCO", "abstract": "abstract_v002"}
        img_extension = {"mscoco": "jpg", "abstract": "png"}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.sentences = True
        if(self.sentences):
            with open('demonstartion_t5.json') as f:
                demonstrations = json.load(f)
            model_name_t5 = "google/t5-v1_1-small"
            model_t5 = T5ForConditionalGeneration.from_pretrained(model_name_t5).to(device)
            tokenizer_t5 = T5Tokenizer.from_pretrained(model_name_t5, legacy=False)
        

        with open(os.path.join(script_dir,'Annotations/MultipleChoice_'+imagesType+'_'+ name + '2014_questions.json'), 'r') as question_file:
            with open(os.path.join(script_dir,'Annotations/'+imagesType+'_'+ name + '2014_annotations.json'), 'r') as answer_file:
                question_data = json.load(question_file)
                answer_data = json.load(answer_file)

                annntoations = (answer_data['annotations'])
                questions = (question_data['questions'])
                
                with h5py.File(path + name + '_embeddings.h5', 'a') as hf:
                    for idx, question in enumerate(tqdm(questions[:length], desc="Preprocessing Images")):
                        if idx == 0:
                            image_features_ds = hf.create_dataset('image_features', shape=(length, 512))#, maxshape=(length, 512))
                            question_features_ds = hf.create_dataset('question_features', shape=(length, 512))#, maxshape=(length, 512))
                            answer_features_ds = hf.create_dataset('answer_features', shape=(length, 18, 512))#, maxshape=(length, 18, 512))
                            correct_answers_ds = hf.create_dataset('correct_answers', shape=(length, 1))#, maxshape=(length, 1))
                            answer_types_ds = hf.create_dataset('answer_types', shape=(length, 1))

                            
                        image_id = question['image_id']
                        question_text = question['question']
                        question_id = question['question_id']
                        answers_text = question['multiple_choices']
                        
                        # find quesion id and image id in annotations
                        for annotation in annntoations:
                            if annotation['question_id'] == question_id and annotation['image_id'] == image_id:
                                correct_answer = annotation['multiple_choice_answer']
                                answer_type = annotation['answer_type']
                                # get the index of the answer in the list of answers
                                for i, possible_answer in enumerate(answers_text):
                                    if possible_answer == correct_answer:
                                        index = i
                                        break
                        
                        if self.sentences:
                            for start_type in demonstrations:
                                if question_text.lower().startswith(start_type):
                                    demostrations_for_question = demonstrations[start_type]
                                    generated_template = generate_masked_template_given_demonstrations(question_text, demostrations_for_question, model_t5, tokenizer_t5)
                                    try:
                                        generated_template = generated_template.split(".")[0]+"."
                                        #if start_type=="is" or start_type=="are":
                                        if "[mask]" in generated_template:
                                            #print(question_text, generated_template)
                                            answers_text = [generated_template.replace("[mask]", f"{answer}") for answer in answers_text]#try except    
                                        else:
                                            break
                                    except:
                                        break
                                    break
                        #print(answers_text)
                        correct_answers = index
                        
                        question_tokens = clip.tokenize([question_text]).to(device)
                        answer_tokens = clip.tokenize(answers_text).to(device)
                        
                        
                        # add leading zeros to image id
                        image_id = str(image_id).zfill(12)

                        img = Image.open(os.path.join(script_dir, "Images", imagesType, imgname_start[imagesType] + "_" + name + year[imagesType] + "_{}."+img_extension[imagesType]).format(image_id))
                        if mode == "scale":
                            img = img.resize((400, 400), Image.Resampling.LANCZOS)
                            image = preprocess(img).unsqueeze(0).to(device)
                        elif mode == "crop":
                            image = preprocess(img).unsqueeze(0).to(device)
                        elif mode == "blackbars":
                            #resize but keep aspect ratio
                            width, height = img.size
                            if width > height:
                                new_width = 400
                                new_height = int(height * (400 / width))
                            else:
                                new_height = 400
                                new_width = int(width * (400 / height))
                            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            # add black bars
                            background = Image.new('RGB', (400, 400), (0, 0, 0))
                            offset = (int((400 - new_width) / 2), int((400 - new_height) / 2))
                            background.paste(img, offset)
                            image = preprocess(background).unsqueeze(0).to(device)

                        else:
                            raise ValueError("mode must be scale, crop or blackbars")
                        
                        image_features = model.encode_image(image)
                        question_features = model.encode_text(question_tokens)
                        answer_features = model.encode_text(answer_tokens)

                        self.path = path
                        # store with h5py
                        image_features_ds[idx] = image_features.detach().cpu().numpy()
                        question_features_ds[idx] = question_features.detach().cpu().numpy()
                        answer_features_ds[idx] = answer_features.detach().cpu().numpy()
                        correct_answers_ds[idx] = correct_answers
                        answer_types_ds[idx] = conversion_answer_type[answer_type]

        self.file = h5py.File(path +  name + '_embeddings.h5', 'r')
        self.image_features = self.file['image_features']
        self.question_features = self.file['question_features']
        self.answer_features = self.file['answer_features']
        self.correct_answers = self.file['correct_answers']
        self.answer_types = self.file['answer_types']

    def load(self, path, device, name="train", length=100):
        # load precalculated features and correct answers from a file
        # only load a subset of the data
        self.device = device
        self.path = path
        self.file = h5py.File(path + name + '_embeddings.h5', 'r')
        self.image_features = self.file['image_features'][:length]
        self.question_features = self.file['question_features'][:length]
        self.answer_features = self.file['answer_features'][:length]
        self.correct_answers = self.file['correct_answers'][:length]
        self.answer_types = self.file['answer_types'][:length]                 

    def __len__(self):
        return len(self.correct_answers)

    def __getitem__(self, index):
        image_features = torch.from_numpy(self.image_features[index]).to(self.device)
        question_features = torch.from_numpy(self.question_features[index]).to(self.device)
        answer_features = torch.from_numpy(self.answer_features[index]).to(self.device)
        correct_answers = torch.from_numpy(self.correct_answers[index]).to(self.device)
        answer_types = torch.from_numpy(self.answer_types[index]).to(self.device)
        return  image_features, answer_features, question_features, correct_answers, answer_types