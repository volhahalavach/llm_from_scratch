import torch
import data.tokens_converter

def generate_next_tokens(model, idx, max_new_token, context_size):
  for _ in range(max_new_token):
    idx_context = idx[:, -context_size:]
    with torch.no_grad():
      logits = model(idx_context)

    logits = logits[:, -1, :]
    # converting logits into probabilities
    probas = torch.softmax(logits, dim=-1)
    # choosing the one with the highest probability - it would be the next
    # token into the sequence
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    # add new token to the rest
    idx = torch.cat((idx, idx_next), dim=1)

  return idx

def generate_and_print_text(model, tokenizer, device, start_context):
  model.eval()  # switch model into evaluation mode
  context_size = model.embeddings.position_embedding.weight.shape[0]
  encoded = data.tokens_converter.text_to_token_ids(start_context, tokenizer).to(device)
  with torch.no_grad():
    token_ids = generate_next_tokens(model=model, idx=encoded, max_new_token=50, context_size=context_size)
    decoded_text = data.tokens_converter.token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
  model.train() # switch back into training mode
