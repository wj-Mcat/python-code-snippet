import torch

def entropy(p):
    """ 
    计算变量的熵
    Compute the entropy of a probability distribution 
    """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


def compute_label():
    import torch
    a = torch.tensor([0, 2, -1])
    b = torch.tensor([1,0,1])
    c = torch.zeros_like(b).\
        index_fill(dim=-1, index=a.index_select(dim=-1, index=torch.where(a!=-1)[0]), value=1)
    # torch.where(a!=-1)
    assert c.item() == b.item()