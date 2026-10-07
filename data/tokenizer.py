import tiktoken

class GPTTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text): #, allowed_special=None): - let's remove it for a while
        return self.tokenizer.encode(text)#, allowed_special)

    def decode(self, ids):
        return self.tokenizer.decode(ids)