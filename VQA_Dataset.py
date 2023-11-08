import datasets
import torch
import clip
import json
from PIL import Image


class VQA_Dataset(datasets.Dataset):
    
    def __init__(self, preprocess, device):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.image_ids = []
        self.question_tokens = []
        self.answer_tokens = []
        
        with open('MultipleChoice_abstract_v002_val2015_questions.json', 'r') as f:
            data = json.load(f)
            questions = (data['questions'])

            
            for idx, question in enumerate(questions):
                self.image_ids.append(question['image_id'])
                self.question_tokens.append(clip.tokenize([question['question']]).to(device))
                self.answer_tokens.append(clip.tokenize(question['multiple_choices']).to(device))  
                
                if idx == 1:
                    print(question) 
                    print(self.image_ids[1])
                    print(self.question_tokens[1])
                    print(self.answer_tokens[1])
        
            # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
        



    def __len__(self):
        return len(self.questions)

    def __getitem__(self, indices):
        return self.images[indices], self.question_tokens[indices], self.answer_tokens[indices]
    

if __name__ == "__main__":

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    dataset = VQA_Dataset(preprocess, device)