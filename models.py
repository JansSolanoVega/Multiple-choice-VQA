import torch

class VQA_Model(torch.nn.Module):
    """uses the model from clip like in evaluation method"""
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device

    def forward(self, image, question_tokens, answer_tokens):
        """returns the logits for the answers"""
        image_features = self.model.encode_image(image)
        question_features = self.model.encode_text(question_tokens)
        answer_features = self.model.encode_text(answer_tokens)
        
        answer_features /= answer_features.norm(dim=-1, keepdim=True)
        
        combined_features = image_features * question_features
        combined_features /= combined_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * combined_features @ answer_features.T).softmax(dim=-1)
        return similarity

    

class VQA_Model2(torch.nn.Module):
    """architecture that uses clip and a small NN to combine question and image features to get a embedding of the same size as the answer embedding
        Only train the NN, not the clip model
    """
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.fc1 = torch.nn.Linear(1024, 512).to(self.device)
        self.initialize_parameters()

    def initialize_parameters(self):
        # Apply Xavier/Glorot initialization to the linear layer
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias.data)

    def forward(self, image, question_tokens, answer_tokens):
        """returns the logits for the answers"""
        image_features = self.model.encode_image(image)

        question_features = self.model.encode_text(question_tokens)

        answer_features = torch.stack([self.model.encode_text(batch_answers_tokens) for batch_answers_tokens in answer_tokens]).to(torch.float32)
        
        answer_features /= answer_features.clone().norm(dim=-1, keepdim=True)

        # concatenate the features
        combined_features = torch.cat((image_features, question_features), dim=1).to(self.device)  
        combined_features /= combined_features.clone().norm(dim=-1, keepdim=True) 
        combined_features = combined_features.to(torch.float32)
        combined_features = self.fc1(combined_features)
        
        # here normalization?
        if self.device == 'cpu':
            similarity = torch.einsum("bn,bqn->bq", [combined_features, answer_features])
        else:
            similarity = (100*torch.einsum("bn,bqn->bq", [combined_features, answer_features])) #scaling before softmax
        
        return similarity

class VQA_Model3(torch.nn.Module):
    """architecture that uses clip and a small NN to combine question and image features to get a embedding of the same size as the answer embedding
        Only train the NN, not the clip model
        with relu activation
    """
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.fc1 = torch.nn.Linear(1024, 512).to(self.device)
        self.initialize_parameters()

    def initialize_parameters(self):
        # Apply Xavier/Glorot initialization to the linear layer
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias.data)

    def forward(self, image, question_tokens, answer_tokens):
        """returns the logits for the answers"""
        image_features = self.model.encode_image(image)

        question_features = self.model.encode_text(question_tokens)

        answer_features = torch.stack([self.model.encode_text(batch_answers_tokens) for batch_answers_tokens in answer_tokens]).to(torch.float32)
        
        answer_features /= answer_features.clone().norm(dim=-1, keepdim=True)

        # concatenate the features
        combined_features = torch.cat((image_features, question_features), dim=1).to(self.device)  
        combined_features /= combined_features.clone().norm(dim=-1, keepdim=True) 
        combined_features = combined_features.to(torch.float32)
        combined_features = self.fc1(combined_features)
        combined_features = torch.nn.functional.relu(combined_features)
        
        # here normalization?
        if self.device == 'cpu':
            similarity = torch.einsum("bn,bqn->bq", [combined_features, answer_features])
        else:
            similarity = (100*torch.einsum("bn,bqn->bq", [combined_features, answer_features])) #scaling before softmax
        
        return similarity

class VQA_Model4(torch.nn.Module):
    """architecture that uses clip and a small NN to combine question and image features to get a embedding of the same size as the answer embedding
        Only train the NN, not the clip model
        with relu activation
    """
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.fc1 = torch.nn.Linear(1024, 512).to(self.device)
        self.fc2 = torch.nn.Linear(512, 512).to(self.device)
        self.initialize_parameters()

    def initialize_parameters(self):
        # Apply Xavier/Glorot initialization to the linear layer
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias.data)

    def forward(self, image, question_tokens, answer_tokens):
        """returns the logits for the answers"""
        image_features = self.model.encode_image(image)

        question_features = self.model.encode_text(question_tokens)

        answer_features = torch.stack([self.model.encode_text(batch_answers_tokens) for batch_answers_tokens in answer_tokens]).to(torch.float32)
        
        answer_features /= answer_features.clone().norm(dim=-1, keepdim=True)

        # concatenate the features
        combined_features = torch.cat((image_features, question_features), dim=1).to(self.device)  
        combined_features /= combined_features.clone().norm(dim=-1, keepdim=True) 
        combined_features = combined_features.to(torch.float32)
        combined_features = self.fc1(combined_features)
        combined_features = torch.nn.functional.relu(combined_features)
        combined_features = self.fc2(combined_features)
        combined_features = torch.nn.functional.relu(combined_features)
        
        # here normalization?
        if self.device == 'cpu':
            similarity = torch.einsum("bn,bqn->bq", [combined_features, answer_features])
        else:
            similarity = (100*torch.einsum("bn,bqn->bq", [combined_features, answer_features])) #scaling before softmax
        
        return similarity

class VQA_Model_Precalc(torch.nn.Module):
    """Model that uses already calulated features"""
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.fc1 = torch.nn.Linear(1024, 512).to(self.device)
        self.initialize_parameters()

    def initialize_parameters(self):
        # Apply Xavier/Glorot initialization to the linear layer
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias.data)

    def forward(self, image_features, question_features, answer_features):
        """returns the logits for the answers"""
        # concatenate the features
        combined_features = torch.cat((image_features, question_features), dim=1).to(self.device)  
        combined_features /= combined_features.clone().norm(dim=-1, keepdim=True) 
        combined_features = combined_features.to(torch.float32)
        combined_features = self.fc1(combined_features)
        combined_features = torch.nn.functional.relu(combined_features)
        
        # here normalization?
        if self.device == 'cpu':
            similarity = torch.einsum("bn,bqn->bq", [combined_features, answer_features])
        else:
            similarity = (100*torch.einsum("bn,bqn->bq", [combined_features, answer_features])) #scaling before softmax
        
        return similarity
    