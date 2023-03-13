# Implementation of the data synthesis algorithm proposet by Shokri et al.
import numpy as np
from tqdm import tqdm
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MIA_DATA_MAX_ITER = 1000
MIA_DATA_CONF_MIN = 0.8 # min prob cutoff to consider a record member of the class
MIA_DATA_REJ_MAX = 5 # max number of consecutive rejections

def features_generator(n_features: int, dtype: str, rang: tuple = (0, 1)):
    if dtype not in ("bool", "int", "float"):
        raise ValueError("Parameter `dtype` must be 'bool', 'int' or 'float'")

    if dtype == "bool":
        x = np.random.randint(0, 2, n_features)
    if dtype == "int":
        x = np.random.randint(rang[0], rang[1], n_features)
    if dtype == "float":
        x = np.random.uniform(rang[0], rang[1], n_features)
    return x


def feature_randomizer(x: np.ndarray, k: int, dtype: str, rang: tuple):
    idx_to_change = np.random.randint(0, x.shape[1], size=k)

    new_feats = features_generator(k, dtype, rang)

    x[0, idx_to_change] = torch.Tensor(new_feats).to(DEVICE)
    return x


def synthesize(target_model, fixed_cls, k_max, dtype, n_features):

    x_shape = [1] + n_features
    x = features_generator(x_shape, dtype=dtype)  # random record
    x = torch.Tensor(x).to(DEVICE)

    y_c_current = 0  # target model’s probability of fixed class
    n_rejects = 0  # consecutives rejections counter
    k = k_max
    k_min = 1
    
    for _ in range(MIA_DATA_MAX_ITER):
        y = target_model(x)  # query target model
        y = torch.nn.functional.softmax(y, dim=1)
        y_c = y[0, fixed_cls]
        if y_c >= y_c_current:
            if (y_c > MIA_DATA_CONF_MIN) and (fixed_cls == np.argmax(y)):
                return x
            # reset vars
            x_new = x
            y_c_current = y_c
            n_rejects = 0
        else:
            n_rejects += 1
            if n_rejects > MIA_DATA_REJ_MAX:
                k = max(k_min, int(np.ceil(k / 2)))
                n_rejects = 0

        # 在random之前要改形状，random之后再改回来
        x_new = x_new.reshape((1, -1))
        x = feature_randomizer(x_new, k, dtype=dtype, rang=(0, 1))
        x = x.reshape(x_shape)

    return y_c_current


def synthesize_batch(input_shape, target_model, fixed_cls, n_records, k_max, dtype):
    n_features = input_shape
    x_batch_size = [n_records] + n_features
    x_synth = np.zeros(x_batch_size)

    for i in tqdm(range(n_records)):
        j = 0
        while True:  # repeat until synth finds record
            x_vec = synthesize(target_model, fixed_cls, k_max, dtype, n_features)
            print(x_vec)
        x_synth[i, :] = x_vec

    return x_synth
