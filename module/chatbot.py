from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
import re

base_pretrained_model_path = "linhvu/decapoda-research-llama-7b-hf"
fine_tuned_parameters_path = "vik1996/llama2_theplantera-chatbot"

tokenizer = LLaMATokenizer.from_pretrained(base_pretrained_model_path)

pretrained_model = LLaMAForCausalLM.from_pretrained(
    base_pretrained_model_path,
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(pretrained_model, fine_tuned_parameters_path)

def get_response(input_text):
    PROMPT = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n' + input_text + '\n### Response:'
    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].cuda()
    
    generation_config = GenerationConfig(
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.15,
    )
    print("Generating...")
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=128,
    )
    output_text = []
    for s in generation_output.sequences:
        #print(tokenizer.decode(s))
        output_text.append(tokenizer.decode(s))

    
    pattern = re.compile(r'### Response:(.*)', re.DOTALL)


    match = pattern.search(output_text[0])


    response_text = match.group(1).strip() if match else None

    return response_text

