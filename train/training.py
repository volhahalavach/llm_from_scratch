from train import cross_entropy_loss
import torch
import train.generate

#train model
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
  train_losses, val_losses, track_tokens_seen = [], [], []
  tokens_seen, global_step = 0, -1

  for epoch in range(num_epochs):
    model.train()
    for input_batch, target_batch in train_loader:
      optimizer.zero_grad()
      batch_loss = cross_entropy_loss.calc_loss_per_batch(input_batch, target_batch, model, device)
      batch_loss.backward()
      optimizer.step()
      tokens_seen += input_batch.numel()
      global_step +=1

      if global_step % eval_freq == 0:
        train_loss, val_loss = evaluate_model(
            model, train_loader, val_loader, device, eval_iter
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        track_tokens_seen.append(tokens_seen)
        print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

    train.generate.generate_and_print_text(model, tokenizer, device, start_context)

  return train_losses, val_losses, track_tokens_seen


# evaluate model
def evaluate_model(model, train_loader, val_loader, device, eval_iter):

  model.eval() # switches the model into eval mode (dropout disabled, BatchNorm  uses stored running statistics instead of statistics from the current batch.)

  with torch.no_grad():
    train_loss = cross_entropy_loss.calc_loss_on_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss = cross_entropy_loss.calc_loss_on_loader(val_loader, model, device, num_batches=eval_iter)

  model.train() # switch the model back into training mode
  return train_loss, val_loss
