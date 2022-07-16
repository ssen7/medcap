import torch

def check_cuda():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        print(f'There are {device_count} GPU(s) available.')
        for i in range(device_count):
            print('Device name:', torch.cuda.get_device_name(i))
        return device
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        return device


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()