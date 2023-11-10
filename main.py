import torch
import clip
from VQA_Dataset import VQA_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm



def evaluate(model, dataloader):
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        image = batch['image'].to(device)
        text = batch['question'].to(device)
        answer = batch['answer'].to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            correct += (torch.argmax(logits_per_image, dim=1) == answer).sum()
            total += len(answer)
    print("Accuracy: ", correct/total)











if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = VQA_Dataset()
    dataset.load_all(preprocess, length=200)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    

        