import pickle
import pickle
import datasets
import torch
from tqdm import tqdm
import clip
from torch.utils.data import DataLoader
import json
from PIL import Image
import h5py
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import random
import os 
from transformers import T5ForConditionalGeneration, T5Tokenizer

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
        self.correct_answers = []
        self.image_features = []
        self.name = ""
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

                    
                    if image_id == 20067:
                        id_print = str(image_id).zfill(12)
                        img = Image.open("Images_abstract/abstract_v002_val2015_{}.png".format(id_print)).convert('RGB')
                        img.show()
                        print("question: ", question_text)
                        print("answers: ", answers_text)
                    else:
                        continue

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
                    
                    img = Image.open("Images/abstract_v002_val2015_0000000{}.png".format(image_id))
                    self.images.append(preprocess(img.resize((400, 400), Image.Resampling.LANCZOS)).unsqueeze(0).to(device))
                    # can we process all images at once?
                    
    def load(self, preprocess, device , name = 'train', length=100, mode="scale"):
        self.device = device
        self.preprocess = preprocess
        self.name = name
        "load without preprocessing or tokenizing"
        with open('Annotations/MultipleChoice_mscoco_'+ name + '2014_questions.json', 'r') as question_file:
            with open('Annotations/mscoco_'+ name + '2014_annotations.json', 'r') as answer_file:
                question_data = json.load(question_file)
                answer_data = json.load(answer_file)

                annntoations = (answer_data['annotations'])
                questions = (question_data['questions'])
                
                
                for idx, question in enumerate(tqdm(questions[:length], desc="Preprocessing Images")):
                    
                       
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
                    
                    


    def get_random_image(self, name="train"):
            random_idx = random.randint(0, len(self.image_ids))     
            image_id = int(self.image_ids[random_idx])

            image_id = str(image_id).zfill(12)
            img = Image.open("Images_real/" + name + "2014/COCO_" + name + "2014_{}.jpg".format(image_id)).convert('RGB')
            print("image id: ", image_id)
            img.show()
            print("question: ", self.questions[random_idx])
            print("answers: ", self.answers[random_idx])
            return self.questions[random_idx], self.answers[random_idx], self.correct_answers[random_idx]
                    
                    

                    

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
                    
                    img = Image.open("Images/abstract_v002_val2015_0000000{}.png".format(image_id))
                    img = img.resize((400, 400), Image.Resampling.LANCZOS)
                    
                    self.image_features.append(model.encode_image(img).to(device))
                    self.question_features.append(model.encode_text(question_tokens).to(device))
                    self.answer_features.append(model.encode_text(answer_tokens).to(device))
                    
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
            if self.preprocess != None:
                img = Image.open("Images_abstract/abstract_v002_val2015_0000000{}.png".format(image_id))
            
                image = preprocess(img.resize((400, 400), Image.Resampling.LANCZOS)).unsqueeze(0).to(device)
                answer_tokens = clip.tokenize(self.answers[index]).to(self.device)
                question_tokens = clip.tokenize([self.questions[index]]).to(self.device)
                return  image, answer_tokens, question_tokens, self.correct_answers[index]
            else:
                image_id = str(image_id).zfill(12)
                img = Image.open("Images_real/" + self.name + "2014/COCO_" + self.name + "2014_{}.jpg".format(image_id)).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
                    transforms.ToTensor()
                    ]) 
                
                image = transform(img).unsqueeze(0).to(self.device)
                return  image, self.answers[index], self.questions[index], self.correct_answers[index]
    
class VQA_Dataset_preloaded(torch.utils.data.Dataset):
    
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
                                    generated_template = generate_masked_template(question_text, demostrations_for_question, model_t5, tokenizer_t5)
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
'''
class VQA_Dataset_BLIP_preloaded(torch.utils.data.Dataset):
    
    def __init__(self):
        self.device = None
        self.file = None
        self.image_features = None
        self.question_features = None
        self.answer_features = None
        self.correct_answers = None
        self.answers = None
        self.questions = None
        self.images = None



        
    def compute_store(self, image_question_model, answer_model, device, path, name="train",length=100, mode="base", each = False):
        """
        preprocess images and tokenize questions and answers and compute embeddings
        store them in a file
        """
        self.device = device
        with open('Annotations/MultipleChoice_mscoco_'+ name + '2014_questions.json', 'r') as question_file:
            with open('Annotations/mscoco_'+ name + '2014_annotations.json', 'r') as answer_file:
                question_data = json.load(question_file)
                answer_data = json.load(answer_file)

                annntoations = (answer_data['annotations'])
                questions = (question_data['questions'])
                
                with h5py.File(path + name + '_embeddings.h5', 'a') as hf:
                    for idx, question in enumerate(tqdm(questions[:length], desc="Preprocessing Images")):
                        if idx == 0:
                            #image_features_ds = hf.create_dataset('image_features', shape=(length, 512), maxshape=(length, 512))
                            image_question_features_ds = hf.create_dataset('image_question_features', shape=(length, 768), maxshape=(length, 768))
                            answer_features_ds = hf.create_dataset('answer_features', shape=(length, 18, 768), maxshape=(length, 18, 768))
                            correct_answers_ds = hf.create_dataset('correct_answers', shape=(length, 1), maxshape=(length, 1))

                            
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
                            
                        correct_answers = index
                        
                        
                        
                        # add leading zeros to image id
                        image_id = str(image_id).zfill(12)
                        img = Image.open("Images_real/" + name + "2014/COCO_" + name + "2014_{}.jpg".format(image_id)).convert('RGB')
                        transform = transforms.Compose([
                            transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
                            transforms.ToTensor()
                            ]) 
                        
                        image = transform(img).unsqueeze(0).to(self.device)
                        
                        if mode == "base":
                            question_features = image_question_model(image, question_text, mode='multimodal').to(device)
                            if each:
                                answer_features = torch.zeros((18,768)).to(device)
                                for c, answer in enumerate(answers_text):
                                    answer_features[c] = answer_model(image, answer, mode='text').squeeze(0).to(device)
                            else:
                                answer_features = answer_model(image, answers_text, mode='text').to(device)
                        elif mode == "itm":
                            # only 1 model is used
                            question_features = image_question_model(image, question_text, answers_text).to(device)
                        

                        self.path = path
                        # store with h5py
                        #image_features_ds[idx] = image.detach().cpu().numpy()
                        image_question_features_ds[idx] = question_features.detach().cpu().numpy()
                        answer_features_ds[idx] = answer_features.detach().cpu().numpy()
                        correct_answers_ds[idx] = correct_answers

        self.file = h5py.File(path +  name + '_embeddings.h5', 'r')
        #self.image_features = self.file['image_features']
        self.question_features = self.file['image_question_features']
        self.answer_features = self.file['answer_features']
        self.correct_answers = self.file['correct_answers']

    def load(self, path, device, name="train", length=100):
        # load precalculated features and correct answers from a file
        # only load a subset of the data
        self.device = device
        self.path = path
        self.file = h5py.File(path + name + '_embeddings.h5', 'r')
        #self.image_features = self.file['image_features'][:length]
        self.question_features = self.file['image_question_features'][:length]
        self.answer_features = self.file['answer_features'][:length]
        self.correct_answers = self.file['correct_answers'][:length]
                    

    def __len__(self):
        return len(self.correct_answers)

    def __getitem__(self, index):
        #image_features = torch.from_numpy(self.image_features[index]).to(self.device)
        image_question_features = torch.from_numpy(self.question_features[index]).to(self.device)
        answer_features = torch.from_numpy(self.answer_features[index]).to(self.device)
        correct_answers = torch.from_numpy(self.correct_answers[index]).to(self.device)
        return  answer_features, image_question_features, correct_answers
'''   
class  VQA_Dataset_Sentences(torch.utils.data.Dataset):
    def __init__(self):
        self.image_ids = []
        self.question_ids = []
        self.questions = []
        self.answers = []
        self.correct_answers = []
        self.answer_types = []
        
    def load(self,preprocess, device, name="val", length=100):
        # load precalculated features and correct answers from a json file
        # only load a subset of the data
        self.device = device
        self.preprocess = preprocess
        self.name = name
        if name == "train":
            file = 'sentences_train.json'
        elif name == "val":
            file = 'sentences_val.json'
        conversion_answer_type = {"yes/no": 0, "number": 1, "other": 2}

        with open(file, 'r') as f:
            data = json.load(f)
            data = data['data']
            for pair in tqdm(data[:length]):
                self.image_ids.append(pair['image_id']) 
                self.question_ids.append(pair['question_id'])
                self.questions.append(pair['question'])
                self.answers.append(pair['answers'])
                self.correct_answers.append(pair['correct_answer'])
                with open('Annotations/mscoco_'+ self.name+ '2014_annotations.json', 'r') as answer_file:
                    answer_data = json.load(answer_file)
                    annntoations = (answer_data['annotations'])
                    for annotation in annntoations:
                        if annotation['question_id'] == pair['question_id'] and annotation['image_id'] == pair['image_id']:
                            answer_type = annotation['answer_type']
                            break
                    self.answer_types.append(conversion_answer_type[answer_type])
                
    def __len__(self):
        return len(self.correct_answers)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        # open image
        image_id = str(image_id).zfill(12)
        img = Image.open("Images/mscoco/"+ "COCO_" + self.name + "2014_{}.jpg".format(image_id))
        image = self.preprocess(img.resize((400, 400), Image.Resampling.LANCZOS)).to(self.device)
        answer_tokens = clip.tokenize(self.answers[index]).to(self.device)
        question_tokens = clip.tokenize([self.questions[index]]).to(self.device)

        return  image, answer_tokens, question_tokens, self.correct_answers[index], self.answer_types[index]

def generate_masked_template(question, demonstrations, model, tokenizer):
    # Construct input with explicit instruction to generate a template
    input_text = f"{question}<extra_id_0>.{' '.join(demonstrations)}"
    # Tokenize the input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()

    # Generate output by filling in the <extra_id_0> token
    output_ids = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

    # Decode the generated output
    generated_template = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # only until first [mask]
    generated_template = generated_template.split("[mask]")[0] + "[mask]."

    return generated_template

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = VQA_Dataset()
    dataset.load_preprocess(preprocess, length=200, device=device)
    dataset.load_preprocess(preprocess, length=200, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i in tqdm(dataloader, desc="Testing"):
        id = i['image_id']
        if id == torch.tensor(26216):
            print("id: ", id, "question: ", i['question'], "Answer: ", i['correct_answer_text'])
            print("Possible answers: ", i['possible_answers'])
            break
        