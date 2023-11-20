import pickle
import datasets
import torch
from tqdm import tqdm
import clip
from torch.utils.data import DataLoader
import json
from PIL import Image


class VQA_Dataset(torch.utils.data.Dataset):
    
    def __init__(self):
        
        self.images = []
        self.questions = []
        self.answers = []
        self.image_ids = []
        self.question_tokens = []
        self.answer_tokens = []  
        self.correct_answers = []
        self.image_features = []
        self.question_features = []
        self.answer_features = []
        self.device = None 
        self.preprocess = None
            # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
        
    def load_preprocess(self, preprocess, device, length=100):
        
        with open('Annotations/MultipleChoice_abstract_v002_val2015_questions.json', 'r') as question_file:
            with open('Annotations/abstract_v002_val2015_annotations.json', 'r') as answer_file:
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
                        
                    self.correct_answers.append(index)
                    self.image_ids.append(torch.tensor(image_id).unsqueeze(0))
                    self.questions.append(question_text)
                    self.answers.append(answers_text)
                    
                    self.question_tokens.append(clip.tokenize([question_text]).to(device))
                    self.answer_tokens.append(clip.tokenize(answers_text).to(device)) 
                    
                    self.images.append(preprocess(Image.open("Images/abstract_v002_val2015_0000000{}.png".format(image_id))).to(device))
                    # can we process all images at once?
                    
    def load(self, preprocess, device , length=100):
        self.device = device
        self.preprocess = preprocess
        "load without preprocessing or tokenizing"
        with open('Annotations/MultipleChoice_abstract_v002_val2015_questions.json', 'r') as question_file:
            with open('Annotations/abstract_v002_val2015_annotations.json', 'r') as answer_file:
                question_data = json.load(question_file)
                answer_data = json.load(answer_file)

                annntoations = (answer_data['annotations'])
                questions = (question_data['questions'])

                for question in tqdm(questions[:length], desc="Loading Images"):
                    
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
                        
                    self.correct_answers.append(index)
                    self.image_ids.append(torch.tensor(image_id).unsqueeze(0))
                    self.questions.append(question_text)
                    self.answers.append(answers_text)
                    
                    # can we process all images at once?

    def load_encode(self, preprocess, device , model, length=100):
        self.device = device
        # preprocess the image and tokenize the question and answers and embbed them
        with open('Annotations/MultipleChoice_abstract_v002_val2015_questions.json', 'r') as question_file:
            with open('Annotations/abstract_v002_val2015_annotations.json', 'r') as answer_file:
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
                        
                    self.correct_answers.append(index)
                    
                    question_tokens = clip.tokenize([question_text]).to(device)
                    answer_tokens = clip.tokenize(answers_text).to(device)
                    image = preprocess(Image.open("Images/abstract_v002_val2015_0000000{}.png".format(image_id))).unsqueeze(0).to(device)

                    
                    self.image_features.append(model.encode_image(image).to("cpu"))
                    self.question_features.append(model.encode_text(question_tokens).to("cpu"))
                    self.answer_features.append(model.encode_text(answer_tokens).to("cpu"))
                    
    def save(self, path):
        # save precalculated features and correct answers in a file
        if len(self.image_features) != 0:
            with open(path + 'embeddings.pkl', "wb") as fOut:
                pickle.dump({'image_features': self.image_features,'answer_features': self.answer_features, 'question_features': self.question_features, 'correct_anwsers': self.correct_answers }, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    def load_saved(self, path):
        # load precalculated features and correct answers from a file
        with open(path + 'embeddings.pkl', "rb") as fIn:
            data = pickle.load(fIn)
            self.image_features = data['image_features']
            self.answer_features = data['answer_features']
            self.question_features = data['question_features']
            self.correct_answers = data['correct_anwsers']

    def __len__(self):
        return len(self.correct_answers)

    def __getitem__(self, index):
        if len(self.images) != 0: 
            return  self.images[index], self.answer_tokens[index], self.question_tokens[index], self.correct_answers[index]      
        elif len(self.image_features) != 0:
            return  self.image_features[index], self.answer_features[index], self.question_features[index], self.correct_answers[index]
        else:
            # preprocess the image and tokenize the question and answers
            image_id = int(self.image_ids[index])
            image = self.preprocess(Image.open("Images/abstract_v002_val2015_0000000{}.png".format(image_id))).to(self.device)
            answer_tokens = clip.tokenize(self.answers[index]).to(self.device)
            question_tokens = clip.tokenize([self.questions[index]]).to(self.device)
            return  image, answer_tokens, question_tokens, self.correct_answers[index]
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = VQA_Dataset()
    dataset.load_preprocess(preprocess, length=200, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i in tqdm(dataloader, desc="Testing"):
        id = i['image_id']
        if id == torch.tensor(26216):
            print("id: ", id, "question: ", i['question'], "Answer: ", i['correct_answer_text'])
            print("Possible answers: ", i['possible_answers'])
            break
        