def convert_ids_to_string(ids, model):
    list_special_tokens = model.tokenizer.all_special_ids
    filtered_ids = [id for id in ids.numpy()[0] if id not in list_special_tokens]

    string_out = model.tokenizer.convert_tokens_to_string(model.tokenizer.convert_ids_to_tokens(filtered_ids))

    return string_out

def convert_string_to_ids(string, model):  
    correct_answer = model.tokenizer(string, padding='max_length', truncation=True, max_length=20, return_tensors="pt")
    correct_answer.input_ids[:,0] = model.tokenizer.enc_token_id
    return correct_answer

def convert_id_to_answer_type(id):
    list = ["yes/no", "number", "other"] 
    return list[int(id)]