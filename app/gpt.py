from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

import torch.nn.functional as F

def generate_with_stable_diffusion(prompt, temperature=1.0, max_length=100):
    # Encode the prompt and get initial input tensor
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate the sequence with temperature control
    with torch.no_grad():
        for _ in range(max_length - len(prompt)):
            logits = model(input_ids).logits[:, -1, :]
            
            # Apply temperature to logits (this is where the stable diffusion happens)
            logits /= temperature

            # Sample next token from the softmax distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Append the next token to the input
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    # Decode the token IDs to string
    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return output


prompt = "The universe is"
result = generate_with_stable_diffusion(prompt, temperature=0.8, max_length=50)
print(result)