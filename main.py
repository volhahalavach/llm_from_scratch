import data.dataloader
from data.tokenizer import GPTTokenizer
import datasets
import config
import model.gpt_model
import torch
import train.training
from analyse import  plot


#from datasets import list_datasets, load_dataset


def load_text():

    dataset = datasets.load_dataset("karpathy/tiny_shakespeare", trust_remote_code=True)
    txt = dataset["train"]['text']
    return txt[0]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_text = load_text()
    #with open("the-verdict.txt", "r", encoding='utf-8') as f:
    #   input_text = f.read()
    print("Total number of characters: ", len(input_text))

    train_text = input_text[:int(len(input_text)*0.8)]
    val_text = input_text[int(len(input_text)*0.8):]
    config = config.GPTConfig()
    tokenizer = GPTTokenizer()
    batch_size = 8

    train_dataloader = data.dataloader.create_data_loader(
        train_text,
        tokenizer,
        config.context_length,
        batch_size,
        shuffle=False,
        drop_last=False
    )

    print(len(train_dataloader))

    val_dataloader = data.dataloader.create_data_loader(
        val_text,
        tokenizer,
        config.context_length,
        batch_size,
        shuffle=False,
        drop_last=False
    )
    print(len(val_dataloader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    my_model = model.gpt_model.GPTModel(config).to(device)
    optimizer = torch.optim.Adam(my_model.parameters())
    train_loss, val_loss, tokens_seen = train.training.train_model(
        my_model,
        train_dataloader,
        val_dataloader,
        optimizer,
        num_epochs=2,
        device=device,
        eval_freq=5,
        eval_iter=1,
        start_context="Love is",
        tokenizer=tokenizer
    )

    epochs_tensor = torch.linspace(0, 2, len(train_loss))
    plot.plot_losses(epochs_tensor, train_loss, val_loss, tokens_seen)







    """tokenizer = GPTTokenizer()
    config = config.GPTConfig()
    dataset = GPTDataset(input_text, tokenizer, config.context_length, config.context_length)
    batch_size = 8
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(inputs.shape)
    print(targets.shape)

    embeddings = GPTEmbedding(config)
    idx = embeddings(inputs)
    print(idx.shape)
    print(idx)"""



