import datasets
import torch
from tqdm import tqdm
import clip
from torch.utils.data import DataLoader
import json
from PIL import Image


class VQA_Dataset(datasets.Dataset):
    
    def __init__(self):
        
        self.images = []
        self.questions = []
        self.answers = []
        self.image_ids = []
        self.question_tokens = []
        self.answer_tokens = []    
        
            # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
        
    def load_all(self, preprocess, length=100):
        
        with open('MultipleChoice_abstract_v002_val2015_questions.json', 'r') as f:
            data = json.load(f)
            questions = (data['questions'])
            for question in tqdm(questions[:length], desc="Preprocessing Images"):
                
                image_id = question['image_id']
                question_text = question['question']
                answer = question['multiple_choices']

                self.image_ids.append(torch.tensor(image_id).unsqueeze(0))
                self.questions.append(question_text)
                self.answers.append(answer)

                self.question_tokens.append(clip.tokenize([question_text]).to(device))
                # self.question_tokens.append(processor(text=[question_text], return_tensors="pt", padding=True))
                self.answer_tokens.append(clip.tokenize(answer).to(device))  
                # self.answer_tokens.append(processor(text=answer, return_tensors="pt", padding=True))

                self.images.append(preprocess(Image.open("Images/abstract_v002_val2015_0000000{}.png".format(image_id))).unsqueeze(0).to(device))
                # can we process all images at once?
                #self.images.append(processor(images=Image.open("Images/abstract_v002_val2015_0000000{}.png".format(image_id)) ,return_tensors="pt", padding=True))
                


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, indices):
            index = indices[0]
            return {
                'image': self.images[index],
                'question_tokens': self.question_tokens[index],
                'answer_tokens': self.answer_tokens[index],
                'image_id': self.image_ids[index]
            }
                #'question': self.questions[index],
                #'answer': self.answers[index]
                # cant return question, answer directly because they are strings and not tensors
           
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    dataset = VQA_Dataset()
    dataset.load_all(preprocess, length=50)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    found = False
    for i in tqdm(dataloader, desc="Testing"):
        id = i['image_id']
        if id == torch.tensor(26216) and not found:
            found = True
            print("id: ", id, "question: ", i['question_tokens'], "answer: ", i['answer_tokens'])
        