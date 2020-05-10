from transformers import AutoModel,AutoTokenizer,AutoModelWithLMHead
import torch,logging


print("loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained("openai-gpt",cache_dir = "/Users/wujingjing/Documents/nlp_models/")
print("loading model ...")
model = AutoModelWithLMHead.from_pretrained("openai-gpt",cache_dir = "/Users/wujingjing/Documents/nlp_models/")
input_context = "the dog"
input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)
outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5) 
for i in range(3): #  3 output sequences were generated
    print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
