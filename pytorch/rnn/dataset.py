from torch.utils import data

class ATIS(data.Dataset):
    """
    一个非常简单的Dataset示例代码
    """
    def __init__(self, X, slots, intents):
        self.X = X
        self.slots = slots
        self.intents = intents
        self.size = X.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx], self.slots[idx], self.intents[idx]
