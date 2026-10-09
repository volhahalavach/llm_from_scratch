import torch
from model.gpt_model import GPTModel
from config import GPTConfig


def save(model, file_name):
    torch.save(model.state_dict(), file_name)

def load(file_name):
    model_uploaded = GPTModel(GPTConfig())
    model_uploaded.load_state_dict(torch.load(file_name))
    model_uploaded.eval()   #switches the model to evaluation mode for inference, disabling the dropout layers and preventing random drop out of any information during inference
    return model_uploaded


#TODO: to add saving&loading of optimizer