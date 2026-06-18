import torch as th
from torch.autograd import Variable
import numpy as np


def identity(x):
    return x


def entropy(p):
    return -th.sum(p * th.log(p), 1)


def kl_log_probs(log_p1, log_p2):
    return -th.sum(th.exp(log_p1)*(log_p2 - log_p1), 1)


# def index_to_one_hot(index, dim):
#     if isinstance(index, np.int32) or isinstance(index, np.int64):
#         one_hot = np.zeros(dim)
#         one_hot[index] = 1.
#     else:
#         one_hot = np.zeros((len(index), dim))
#         one_hot[np.arange(len(index)), index] = 1.
#     return one_hot

# def index_to_one_hot(index, dim):
#     index = np.asarray(index)
#     if index.ndim == 1:
#         one_hot = np.zeros((index.size, dim))
#         one_hot[np.arange(index.size), index] = 1.
#     elif index.ndim == 2:
#         one_hot = np.zeros((index.shape[0], index.shape[1], dim))
#         for i in range(index.shape[1]):
#             one_hot[np.arange(index.shape[0]), i, index[:, i]] = 1.
#     else:
#         raise ValueError("Index array must be 1 or 2-dimensional")
#     return one_hot

def index_to_one_hot(index, dim):
    index = np.asarray(index)
    if index.ndim == 1 and index.size != 0:
        one_hot = np.zeros((index.size, dim))
        one_hot[np.arange(index.size), index] = 1
    elif index.ndim == 2  and index.size != 0:
        batch_size, seq_length = index.shape
        one_hot = np.zeros((batch_size, seq_length, dim))
        # Efficiently create one-hot encoding using advanced indexing
        batch_indices = np.arange(batch_size)[:, None]  # Shape: (batch_size, 1)
        seq_indices = np.arange(seq_length)[None, :]    # Shape: (1, seq_length)
        one_hot[batch_indices, seq_indices, index] = 1
    elif index.size == 0:
        print(np.arange(index.size), index)
    else:
        raise ValueError("Index array must be 1 or 2-dimensional")
    return one_hot

def to_tensor_var(x, use_cuda, dtype="float"):
    def flatten_dict(d):
        values = []
        for key, value in d.items():
            if isinstance(value, dict):
                values.extend(flatten_dict(value))
            elif isinstance(value, list):
                values.extend(value)
            elif isinstance(value, (int, float)):
                values.append(value)
            else:
                # Handle non-numeric types if necessary
                pass
        return values

    if isinstance(x, dict):
        x = flatten_dict(x)
    
    if dtype == "float":
        x = np.array(x, dtype=np.float32).tolist()
        tensor = th.FloatTensor(x)
    elif dtype == "long":
        x = np.array(x, dtype=np.int64).tolist()
        tensor = th.LongTensor(x)
    elif dtype == "byte":
        x = np.array(x, dtype=np.uint8).tolist()
        tensor = th.ByteTensor(x)
    else:
        x = np.array(x, dtype=np.float64).tolist()
        tensor = th.FloatTensor(x)
    
    if use_cuda:
        tensor = tensor.cuda()
    
    return Variable(tensor)

def agg_double_list(l):
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    return s_mu, s_std