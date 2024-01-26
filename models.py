import torch
import transformers
import torch.nn.functional as F
import os

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

        
        if len(answer_tokens.shape) != 3:
                answer_tokens = answer_tokens.unsqueeze(0)

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
        
    def save(self, path):
        # do not save the clip model as it is not trained
        torch.save(self.fc1.state_dict(), path + 'fc1.pth')
        torch.save(self.fc2.state_dict(), path + 'fc2.pth')

    def load(self, path):
        self.fc1.load_state_dict(torch.load(path + 'fc1.pth'))
        self.fc2.load_state_dict(torch.load(path + 'fc2.pth'))

class VQA_Model1_Precalc(torch.nn.Module):
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
        
        # here normalization?
        if self.device == 'cpu':
            similarity = torch.einsum("bn,bqn->bq", [combined_features, answer_features])
        else:
            similarity = (100*torch.einsum("bn,bqn->bq", [combined_features, answer_features])) #scaling before softmax
        
        return similarity
    
    def save(self, path):
        # do not save the clip model as it is not trained
        path = os.path.join("results_training_CLIP", path)
        torch.save(self.fc1.state_dict(), path + 'fc1.pth')

    def load(self, path):
        path = os.path.join("results_training_CLIP", path)
        self.fc1.load_state_dict(torch.load(path + 'fc1.pth'))

class VQA_Model4_Precalc(torch.nn.Module):
    """Model that uses already calulated features"""
    
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

    def forward(self, image_features, question_features, answer_features):
        """returns the logits for the answers"""

        
        if len(answer_features.shape) != 3:
                answer_features = answer_features.unsqueeze(0)


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
    
    def save(self, path):
        # do not save the clip model as it is not trained
        torch.save(self.fc1.state_dict(), path + 'fc1.pth')
        torch.save(self.fc2.state_dict(), path + 'fc2.pth')

    def load(self, path):
        self.fc1.load_state_dict(torch.load(path + 'fc1.pth'))
        self.fc2.load_state_dict(torch.load(path + 'fc2.pth'))

class VQA_Model_Discr(torch.nn.Module):
    """Model that uses already calulated features"""
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.fc1 = torch.nn.Linear(1024, 768).to(self.device)
        self.fc2 = torch.nn.Linear(768, 512).to(self.device)
        self.initialize_parameters()


    def initialize_parameters(self):
        # Apply Xavier/Glorot initialization to the linear layer
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias.data)

    def forward(self, image_features, question_features):
        """returns the logits for the answers"""
        # concatenate the features
        combined_features = torch.cat((image_features, question_features), dim=1).to(self.device)  
        combined_features /= combined_features.clone().norm(dim=-1, keepdim=True) 
        combined_features = combined_features.to(torch.float32)
        combined_features = self.fc1(combined_features)
        combined_features = torch.nn.functional.relu(combined_features)
        combined_features = self.fc2(combined_features)
        combined_features = torch.nn.functional.relu(combined_features)
        
        return combined_features
    
    def save(self, path):
        # do not save the clip model as it is not trained
        path = os.path.join("results_training_CLIP", path)
        torch.save(self.fc1.state_dict(), path + 'fc1.pth')
        torch.save(self.fc2.state_dict(), path + 'fc2.pth')

    def load(self, path):
        path = os.path.join("results_training_CLIP", path)
        self.fc1.load_state_dict(torch.load(path + 'fc1.pth'))
        self.fc2.load_state_dict(torch.load(path + 'fc2.pth'))

class VQA_Model_Discr_Siamese(torch.nn.Module):
    """Model that uses already calulated features"""
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.fc1 = torch.nn.Linear(1024, 512).to(self.device)
        self.fc2 = torch.nn.Linear(512, 128).to(self.device)
        self.fc3 = torch.nn.Linear(128, 32).to(self.device)
        self.fc4 = torch.nn.Linear(64, 1).to(self.device)
        self.initialize_parameters()


    def initialize_parameters(self):
        # Apply Xavier/Glorot initialization to the linear layer
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias.data)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias.data)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.zeros_(self.fc4.bias.data)

    def siamese(self, x):
        z = torch.nn.functional.relu(self.fc2(x))     # Size([bs, 512])
        z = torch.nn.functional.relu(self.fc3(z))     # Size([bs, 256])
        return z

    def forward(self, image_features, question_features, target_features):
        """returns the logits for the answers"""
        # concatenate the features
        combined_features = torch.cat((image_features, question_features), dim=1).to(self.device)  
        combined_features /= combined_features.clone().norm(dim=-1, keepdim=True) 
        combined_features = combined_features.to(torch.float32)
        combined_features = self.fc1(combined_features)
        combined_features = torch.nn.functional.relu(combined_features)
        
        output1 = self.siamese(combined_features)
        output2 = self.siamese(target_features)

        concat = torch.cat((output1, output2), dim=1).to(self.device)  
        output = self.fc4(concat)
        return output
    
    def save(self, path):
        # do not save the clip model as it is not trained
        path = os.path.join("results_training_CLIP", path)
        torch.save(self.fc1.state_dict(), path + 'fc1.pth')
        torch.save(self.fc2.state_dict(), path + 'fc2.pth')
        torch.save(self.fc3.state_dict(), path + 'fc3.pth')
        torch.save(self.fc4.state_dict(), path + 'fc4.pth')

    def load(self, path):
        path = os.path.join("results_training_CLIP", path)
        self.fc1.load_state_dict(torch.load(path + 'fc1.pth'))
        self.fc2.load_state_dict(torch.load(path + 'fc2.pth'))
        self.fc3.load_state_dict(torch.load(path + 'fc3.pth'))
        self.fc4.load_state_dict(torch.load(path + 'fc4.pth'))

class VQA_Model_Precalc(torch.nn.Module):
    """Model that uses already calulated features"""
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.fc1 = torch.nn.Linear(1024, 896).to(self.device)
        self.fc2 = torch.nn.Linear(896, 768).to(self.device)
        self.fc3 = torch.nn.Linear(768, 512).to(self.device)
        self.initialize_parameters()

        #self.dropout = torch.nn.Dropout(0.2)

        print("ga")


    def initialize_parameters(self):
        # Apply Xavier/Glorot initialization to the linear layer
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias.data)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias.data)

    def forward(self, image_features, question_features, answer_features):
        """returns the logits for the answers"""
        # concatenate the features
        combined_features = torch.cat((image_features, question_features), dim=1).to(self.device)  
        combined_features /= combined_features.clone().norm(dim=-1, keepdim=True) 
        combined_features = combined_features.to(torch.float32)
        combined_features = self.fc1(combined_features)
        combined_features = torch.nn.functional.relu(combined_features)
        combined_features = self.fc2(combined_features)
        combined_features = torch.nn.functional.relu(combined_features)
        combined_features = self.fc3(combined_features)
        
        # here normalization?
        #print(combined_features.shape)
        #print(answer_features.shape)

        # if batch size is 1, einsum does not work (end of val?)
        if len(answer_features.shape) == 3:
            similarity = (100*torch.einsum("bn,bqn->bq", [combined_features, answer_features]))
        else:
            answer_features = answer_features.unsqueeze(0)
            similarity = (100*torch.einsum("bn,bqn->bq", [combined_features, answer_features])) 
        
        return similarity
    
    def save(self, path):
        # do not save the clip model as it is not trained
        path = os.path.join("results_training_CLIP", path)
        torch.save(self.fc1.state_dict(), path + 'fc1.pth')
        torch.save(self.fc2.state_dict(), path + 'fc2.pth')
        torch.save(self.fc3.state_dict(), path + 'fc3.pth')

    def load(self, path):
        path = os.path.join("results_training_CLIP", path)
        self.fc1.load_state_dict(torch.load(path + 'fc1.pth'))
        self.fc2.load_state_dict(torch.load(path + 'fc2.pth'))
        self.fc3.load_state_dict(torch.load(path + 'fc3.pth'))

class VQA_Model_Precalc_Zero(torch.nn.Module):
    """Model that uses already calulated features"""
    def __init__(self, model, device):
        super().__init__()

    def forward(self, image_features, question_features, answer_features):
        """returns the logits for the answers"""
        # combine the features by averaging
        combined_features = (image_features + question_features)/2


        if len(answer_features.shape) == 3:
            similarity = (100*torch.einsum("bn,bqn->bq", [combined_features, answer_features]))
        else:
            answer_features = answer_features.unsqueeze(0)
            similarity = (100*torch.einsum("bn,bqn->bq", [combined_features, answer_features])) 
        
        return similarity
    
class VQA_Model_Precalc_Text(torch.nn.Module):
    """Model that uses already calulated features"""
    def __init__(self, model, device):
        super().__init__()

    def forward(self, image_features, question_features, answer_features):
        """returns the logits for the answers"""
        # combine the features by averaging
        


        if len(answer_features.shape) == 3:
            similarity = (100*torch.einsum("bn,bqn->bq", [question_features, answer_features]))
        else:
            answer_features = answer_features.unsqueeze(0)
            similarity = (100*torch.einsum("bn,bqn->bq", [question_features, answer_features])) 
        
        return similarity
    


class VQA_Model_classify(torch.nn.Module):
    def __init__(self, device, input_size1 = 512, input_size2 = 512, input_size3 = 18 * 512, hidden_size1 = 256, hidden_size2 = 128, num_classes = 18):

        super(VQA_Model_classify, self).__init__()

        # Define the layers for the first input
        self.fc1_1 = torch.nn.Linear(input_size1, hidden_size1).to(device)
        self.relu1_1 = torch.nn.ReLU()

        # Define the layers for the second input
        self.fc1_2 = torch.nn.Linear(input_size2, hidden_size1).to(device)
        self.relu1_2 = torch.nn.ReLU()

        # Define the layers for the third input
        self.fc1_3 = torch.nn.Linear(input_size3, hidden_size1).to(device)
        self.relu1_3 = torch.nn.ReLU()
        self.input_size3 = input_size3

        # Combine the processed inputs
        combined_size = hidden_size1 * 3
        self.fc2 = torch.nn.Linear(combined_size, hidden_size2).to(device)
        self.relu2 = torch.nn.ReLU()

        # Output layer
        self.fc3 = torch.nn.Linear(hidden_size2, num_classes).to(device)

    def forward(self, x1, x2, x3):
        # Forward pass for the first input
        out1 = self.relu1_1(self.fc1_1(x1))

        # Forward pass for the second input
        out2 = self.relu1_2(self.fc1_2(x2))

        # Forward pass for the third input
        if self.input_size3 > 512:
            if len(x3.shape) != 3:
                x3 = x3.unsqueeze(0)

            x3_flat = x3.view(x3.size(0), -1)  # Flatten along the second dimension
        else:
            x3_flat = x3    
        out3 = self.relu1_3(self.fc1_3(x3_flat))

        # Concatenate the processed inputs
        combined = torch.cat((out1, out2, out3), dim=1)

        # Forward pass through the combined layers
        out_combined = self.relu2(self.fc2(combined))

        # Output layer
        out = self.fc3(out_combined)

        return out


    def save(self, path):
        # do not save the clip model as it is not trained
        torch.save(self.fc1_1.state_dict(), path + 'fc1_1.pth')
        torch.save(self.fc1_2.state_dict(), path + 'fc1_2.pth')
        torch.save(self.fc1_3.state_dict(), path + 'fc1_3.pth')
        torch.save(self.fc2.state_dict(), path + 'fc2.pth')
        torch.save(self.fc3.state_dict(), path + 'fc3.pth')

    def load(self, path):
        self.fc1_1.load_state_dict(torch.load(path + 'fc1_1.pth'))
        self.fc1_2.load_state_dict(torch.load(path + 'fc1_2.pth'))
        self.fc1_3.load_state_dict(torch.load(path + 'fc1_3.pth'))
        self.fc2.load_state_dict(torch.load(path + 'fc2.pth'))
        self.fc3.load_state_dict(torch.load(path + 'fc3.pth'))

class VQA_Model_classify_v2(torch.nn.Module):
    def __init__(self, device, input_size1 = 512, input_size2 = 512, input_size3 = 512):

        super(VQA_Model_classify_v2, self).__init__()
        # Define a module for combining representations
        self.combine_layer = torch.nn.Linear(
            input_size1 + input_size2 + input_size3,
            128  # You can adjust the size of the combined representation
        ).to(device)

        # Define a module for the final classification layer for each answer
        self.final_layers = torch.nn.ModuleList([
            torch.nn.Linear(128, 1) for _ in range(18)
        ]).to(device)

    def forward(self, x1, x2, x3):

       # Separate processing for each answer representation
        answer_scores = []
        if len(x3.shape) != 3:
                x3 = x3.unsqueeze(0)
        for i, answer_representation in enumerate(x3.unbind(1)):
            # Process each answer separately
            # Combine img, question, and the current answer representation
            combined_representation = torch.cat([x1, x2, answer_representation], dim=1)

            # Additional processing if needed
            combined_representation = torch.relu(self.combine_layer(combined_representation))

            # Final classification for the current answer
            answer_scores.append(self.final_layers[i](combined_representation))

        # Stack the scores along dimension 1 to get the final output
        output_scores = torch.cat(answer_scores, dim=1)

        return output_scores


    def save(self, path):
        # do not save the clip model as it is not trained
        torch.save(self.combine_layer.state_dict(), path + 'combine_layer.pth')
        for i, layer in enumerate(self.final_layers):
            torch.save(layer.state_dict(), path + 'final_layer_' + str(i) + '.pth')
        


    def load(self, path):
        self.combine_layer.load_state_dict(torch.load(path + 'combine_layer.pth'))
        for i, layer in enumerate(self.final_layers):
            layer.load_state_dict(torch.load(path + 'final_layer_' + str(i) + '.pth'))

class VQA_Model_classify_v3(torch.nn.Module):
    def __init__(self, device, input_size1 = 512, input_size2 = 512, input_size3 = 512, hidden_size= 512):

        super(VQA_Model_classify_v3, self).__init__()
        # Define a module for combining representations
        self.combine_layer = torch.nn.Linear(
            input_size1 + input_size2,
            hidden_size  # You can adjust the size of the combined representation
        ).to(device)

        self.device = device
        # Define a module for the final classification layer for each answer
        self.final_layer = torch.nn.Linear(hidden_size + input_size3, 1).to(device)

    def forward(self, x1, x2, x3):
        combined_features = torch.cat((x1, x2), dim=1).to(self.device)

        #combined_features /= combined_features.clone().norm(dim=-1, keepdim=True)
        combined_features = self.combine_layer(combined_features)
        combined_features = torch.nn.functional.relu(combined_features)

        answer_scores = []
        
        if len(x3.shape) != 3:
                x3 = x3.unsqueeze(0)
        for i, answer_representation in enumerate(x3.unbind(1)):
            # Final classification for the current answer
            combined_representation = torch.cat([combined_features, answer_representation], dim=1)
            answer_scores.append(self.final_layer(combined_representation))
        output_scores = torch.cat(answer_scores, dim=1)
        return output_scores


    def save(self, path):
        # do not save the clip model as it is not trained
        torch.save(self.combine_layer.state_dict(), path + 'combine_layer.pth')
        torch.save(self.final_layer.state_dict(), path + 'final_layer.pth')

    def load(self, path):
        self.combine_layer.load_state_dict(torch.load(path + 'combine_layer.pth'))
        self.final_layer.load_state_dict(torch.load(path + 'final_layer.pth'))



class VQA_Model_Blip(torch.nn.Module):
    """
    Model that uses BERT to encode the question and image together with a BERT model.
    """
    def __init__(self, device, answer_model, image_question_model, embed_dim=512):
        super(VQA_Model_Blip, self).__init__()
        self.answer_model = answer_model
        self.image_question_model = image_question_model
        self.device = device
        
        #self.fc_multi = torch.nn.Linear(answer_model.text_width, embed_dim).to(self.device)
        #self.fc_text = torch.nn.Linear(answer_model.text_width, embed_dim).to(self.device)

    def forward(self, image, question, answers):
        image_question_feature = self.image_question_model(image, question, mode='multimodal')
        #print(image_question_feature.shape )
        
        #image_question_feature = self.fc_multi(image_question_feature)
        
        answers_features = self.answer_model(image, answers, mode='text')
        
        #print(answers_features.shape)

        #answers_features = self.fc_text(answers_features)
        answers_features = answers_features.unsqueeze(0)


        similarity = torch.einsum("bn,bqn->bq", [image_question_feature, answers_features])

        return similarity
    
    def save(self, path):
        # do not save the clip model as it is not trained
        
        torch.save(self.image_question_model.state_dict(), path + 'image_question_model.pth')
        #torch.save(self.fc_multi.state_dict(), path + 'fc_multi.pth')
        #torch.save(self.fc_text.state_dict(), path + 'fc_text.pth')

    def load(self, path):
        self.fc_multi.load_state_dict(torch.load(path + 'fc_multi.pth'))
        self.fc_text.load_state_dict(torch.load(path + 'fc_text.pth'))

class VQA_Model_Blip_Precalc(torch.nn.Module):
    """
    Model that uses BERT to encode the question and image together with a BERT model.
    """
    def __init__(self, device, answer_model, image_question_model, embed_dim=512):
        super(VQA_Model_Blip, self).__init__()
        self.device = device
        
        #self.fc_multi = torch.nn.Linear(answer_model.text_width, embed_dim).to(self.device)
        #self.fc_text = torch.nn.Linear(answer_model.text_width, embed_dim).to(self.device)

    def forward(self, image_question_features, answers_features):
        
        answers_features = answers_features.unsqueeze(0)


        similarity = torch.einsum("bn,bqn->bq", [image_question_features, answers_features])

        return similarity
    
    def save(self, path):
        # do not save the clip model as it is not trained
        
        torch.save(self.image_question_model.state_dict(), path + 'image_question_model.pth')
        #torch.save(self.fc_multi.state_dict(), path + 'fc_multi.pth')
        #torch.save(self.fc_text.state_dict(), path + 'fc_text.pth')

    def load(self, path):
        self.fc_multi.load_state_dict(torch.load(path + 'fc_multi.pth'))
        self.fc_text.load_state_dict(torch.load(path + 'fc_text.pth'))