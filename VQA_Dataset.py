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
            # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
        
    def load_all(self, preprocess, device, length=100):
        
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
                    # self.question_tokens.append(processor(text=[question_text], return_tensors="pt", padding=True))
                    self.answer_tokens.append(clip.tokenize(answers_text).to(device).unsqueeze(0)) 
                    # self.answer_tokens.append(processor(text=answer, return_tensors="pt", padding=True))
                    img = Image.open("Images/abstract_v002_val2015_0000000{}.png".format(image_id))
                    self.images.append(preprocess(img.resize((400, 400), Image.Resampling.LANCZOS)).unsqueeze(0).to(device))
                    #self.images.append(preprocess(Image.open("Images/abstract_v002_val2015_0000000{}.png".format(image_id))).unsqueeze(0).to(device))
                    # can we process all images at once?
                    #self.images.append(processor(images=Image.open("Images/abstract_v002_val2015_0000000{}.png".format(image_id)) ,return_tensors="pt", padding=True))
                    


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
            result = {
                'image': self.images[index],
                'answer_tokens': self.answer_tokens[index],
                'question_tokens': self.question_tokens[index],
                'image_id': self.image_ids[index],
                'correct_answer_idx': self.correct_answers[index],
                'question': self.questions[index],
                'possible_answers': self.answers[index],
                'correct_answer_text': self.answers[index][self.correct_answers[index]],
            }
           
            return result
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = VQA_Dataset()
    dataset.load_all(preprocess, length=200, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i in tqdm(dataloader, desc="Testing"):
        id = i['image_id']
        if id == torch.tensor(26216):
            print("id: ", id, "question: ", i['question'], "Answer: ", i['correct_answer_text'])
            print("Possible answers: ", i['possible_answers'])
            break
        