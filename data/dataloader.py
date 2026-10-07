from torch.utils.data import DataLoader
from data.dataset import GPTDataset

def create_data_loader(text, tokenizer, context_length, batch_size, shuffle=True, drop_last=False):

    dataset = GPTDataset(text, tokenizer, max_length=context_length, stride=context_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader