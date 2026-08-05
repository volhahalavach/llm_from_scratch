from torch.utils.data import DataLoader

import config
from data.dataset import GPTDataset

from data.tokenizer import GPTTokenizer
from model.embeddings import GPTEmbedding
import datasets
import config


#from datasets import list_datasets, load_dataset


def load_text():

    dataset = datasets.load_dataset("karpathy/tiny_shakespeare", trust_remote_code=True)
    txt = dataset["train"]['text']
    return txt[0]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_text = load_text()
    tokenizer = GPTTokenizer()
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
    print(idx)


# Note: I've commented #allowed_special={"<|endoftext|>"}) in GPTDataset - take a look



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
