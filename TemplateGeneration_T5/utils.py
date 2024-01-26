# open demonstration from json file and load them into dict
import json


def generate_masked_template_given_demonstrations(question, demonstrations, model, tokenizer):
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

def generate_template_answer(question, model, tokenizer):
    with open('demonstration_t5.json') as f:
        demonstrations = json.load(f)

    for start in demonstrations:
        if question.lower().startswith(start):
            generated_template = generate_masked_template_given_demonstrations(question, demonstrations["how many"], model, tokenizer)
            break

    # Print the result
    print("Question:", question)
    print("Generated Template:", generated_template)

    return generated_template