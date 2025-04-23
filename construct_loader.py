import torch.utils.data as Data

def construct_loader(data, batch_size, uncom=False):

    if len(data) == 2:
        loader_dataset = Data.TensorDataset(data[0], data[1])
        loader = Data.DataLoader(dataset=loader_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    elif len(data) == 6:
        loader_dataset = Data.TensorDataset(data[0], data[1], data[2], data[3], data[4], data[5])
        loader = Data.DataLoader(dataset=loader_dataset, batch_size=batch_size, shuffle=True)
    elif len(data) == 8:
        loader_dataset = Data.TensorDataset(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])
        loader = Data.DataLoader(dataset=loader_dataset, batch_size=batch_size, shuffle=True)
    elif len(data) == 3:
        loader = []
        for i in range(data):
            loader_dataset = Data.TensorDataset(data[0][i], data[1][i], data[2][i])
            data_loader = Data.DataLoader(dataset=loader_dataset, batch_size=batch_size, shuffle=True)
            loader.append(data_loader)
    if uncom is True:
        loader = []
        for i in range(len(data[0])):
            loader_dataset = Data.TensorDataset(data[0][i], data[1][i], data[2][i], data[3])
            data_loader = Data.DataLoader(dataset=loader_dataset, batch_size=batch_size, shuffle=True)
            loader.append(data_loader)

    return loader