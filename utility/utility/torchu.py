def tensor_to_numpy(x):
    return x.cpu().detach().numpy()