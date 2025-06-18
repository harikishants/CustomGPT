import torch
from tokenizer import BPETokenizer

def generate_text(prompt, model, max_new_tokens=100, temperature=1.0, sampling='prob'):

    tokenizer = BPETokenizer(vocab_file='vocab/vocab_20000')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    context_length = model.hparams['context_length']

    # Tokenize prompt
    # print(f'[info] prompt: {prompt}')
    tokenized = tokenizer.tokenize(prompt)
    # print(f'[info] No. of tokens in prompt = {len(tokenized.ids)}')
    input_ids = tokenized.ids[-context_length:]
    print(f'[info] context: {tokenizer.decode(input_ids)}')
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)  # (1, T)

    response_tokens = []

    # Generate new tokens autoregressively
    for _ in range(max_new_tokens):
        input_trunc = input_ids[:, -context_length:]  # truncate inputs to context length, taking recent tokens
        with torch.no_grad():
            logits = model(input_trunc)  # (1, T, vocab_size)
            logits = logits[:, -1, :] / temperature  # take logits of last token
            probs = torch.softmax(logits, dim=-1)

            if sampling == 'greedy':
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else: # probabilitstic by default
                next_token = torch.multinomial(probs, num_samples=1)  # sampling
                
        response_tokens.append(next_token)
        input_ids = torch.cat((input_ids, next_token), dim=1)  # append to sequence

        # optional: stop if EOS token generated
        # if next_token.item() == tokenizer.get_eos_id():
        #     break

    # Decode token IDs to text
    # generated_ids = input_ids[0].tolist()
    # new_tokens = generated_ids[len(tokenized.ids):]  # removing prompt tokens
    response = tokenizer.decode(response_tokens)
    print(f'[info] response: {response}')
    return response
