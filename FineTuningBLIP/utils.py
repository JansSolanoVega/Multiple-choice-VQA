def convert_ids_to_string(ids, model):
    list_special_tokens = model.tokenizer.all_special_ids
    filtered_ids = [id for id in ids.numpy()[0] if id not in list_special_tokens]

    string_out = model.tokenizer.convert_tokens_to_string(model.tokenizer.convert_ids_to_tokens(filtered_ids))

    return string_out