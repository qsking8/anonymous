import os
import sys
import logging
import random

import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def mean(items):
    return sum(items)/len(items)


def max_with_index(values):
    best_v = values[0]
    best_i = 0
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_v, best_i


def shuffle(*items):
    example, *_ = items
    batch_size, *_ = example.size()
    index = torch.randperm(batch_size, device=example.device)

    return [item[index] for item in items]


def to_device(*items):
    return [item.to(device=device) for item in items]


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, output_directory: str, log_name: str, debug: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.propagate = False
    return logger
    

def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1


def compute_kendall_tau(a, b):
    assert len(a) == len(b), "Sequence a and b should have the same length while computing kendall tau."
    length = len(a)
    count = 0
    total = 0
    for i in range(length-1):
        for j in range(i+1, length):
            count += _sign(a[i] - a[j]) * _sign(b[i] - b[j])
            total += 1
    Ktau = count / total
    return Ktau


def infer(buffer, ranker):
    matrix0 = []
    ops0 = []
    matrix1 = []
    ops1 = []
    for ((matrix0_, ops0_), (matrix1_, ops1_)) in buffer:
        matrix0.append(matrix0_)
        ops0.append(ops0_)

        matrix1.append(matrix1_)
        ops1.append(ops1_)

    matrix0 = torch.stack(matrix0, dim=0)
    ops0 = torch.stack(ops0, dim=0)

    matrix1 = torch.stack(matrix1, dim=0)
    ops1 = torch.stack(ops1, dim=0)

    with torch.no_grad():
        outputs = ranker((matrix0, ops0), (matrix1, ops1))

    return _sign((outputs-0.5).cpu().tolist())


def select(items, index):
    return [items[i] for i in index]


def list_select(items, index):
    return [item[index] for item in items]


def transpose_l(items):
    return list(map(list, zip(*items)))


def index_generate(m, n, up_triangular=False, max_batch_size=1024):
    if up_triangular:
        indexs = []
        for i in range(m-1):
            for j in range(i+1, n):
                indexs.append((i, j))
                if len(indexs) == max_batch_size:
                    yield indexs
                    indexs = []
        if indexs:
            yield indexs
    else:
        indexs = []
        for i in range(m):
            for j in range(n):
                indexs.append((i, j))
                if len(indexs) == max_batch_size:
                    yield indexs
                    indexs = []
        if indexs:
            yield indexs


def batchify(items):
    if isinstance(items[0], (list, tuple)):
        transposed_items = transpose_l(items)
        return [torch.stack(item, dim=0) for item in transposed_items]
    else:
        return torch.stack(items, dim=0) 
    


def cartesian_traverse(arch0, arch1, ranker, up_triangular=False):
    m, n = len(arch0), len(arch1)
    outputs = []
    with torch.no_grad():
        for index in index_generate(m, n, up_triangular):
            i, j = transpose_l(index)
            a = batchify(select(arch0, i))
            b = batchify(select(arch1, j))
            output = ranker(a, b)
            outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    if up_triangular:
        return outputs
    else:
        return outputs.view(m, n)


def compute_kendall_tau_AR(ranker, archs, performances):
    
    length = len(performances)
    count = 0
    total = 0

    archs = transpose_l([torch.unbind(item) for item in archs])
    outputs = cartesian_traverse(archs, archs, ranker, up_triangular=True)

    p_combination = _sign((outputs-0.5).cpu().tolist())

    for i in range(length-1):
        for j in range(i+1, length):
            count += p_combination[total] * _sign(performances[i]-performances[j])
            total += 1

    assert len(p_combination) == total
    Ktau = count / total
    return Ktau


def concat(a, b):
    return [torch.cat([item0, item1]) for item0, item1 in zip(a, b)]


def list_concat(a, b):
    if len(a) == 3:
        a0, a1, a2 = a
        b0, b1, b2 = b
        rev0 = concat(a0, b0)
        rev1 = concat(a1, b1)
        rev2 = torch.cat([a2, b2], dim=0)
        return rev0, rev1, rev2
    else:
        a0, a1 = a
        b0, b1 = b
        rev0 = concat(a0, b0)
        rev1 = concat(a1, b1)
        return rev0, rev1

def compute_flops(module: nn.Module, size, skip_pattern, device):
    
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)
    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size).to(device))
        module.train(mode=training)
        
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d):
            
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops

def compute_nparam(module: nn.Module, skip_pattern):
    n_param = 0
    for name, p in module.named_parameters():
        if skip_pattern not in name:
            n_param += p.numel()
    return n_param


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def generate_mix_data(id_data, ood_data, ood_rate):
    """
    Merge two PyTorch datasets with a specified out-of-distribution rate.

    Args:
        id_data (torch.utils.data.Dataset): PyTorch dataset containing in-distribution samples.
        ood_data (torch.utils.data.Dataset): PyTorch dataset containing out-of-distribution samples.
        ood_rate (float): The rate of out-of-distribution samples relative to in-distribution samples.

    Returns:
        torch.utils.data.Dataset: Merged dataset with a flag indicating the origin of each sample.
    """
    
    ood_num_samples = int(len(id_data) * ood_rate)

    
    
    ood_subset_indices = random.sample(range(len(ood_data)), min(ood_num_samples, len(ood_data)))
    ood_subset = Subset(ood_data, ood_subset_indices)
    
    
    id_flag = torch.zeros(len(id_data), dtype=torch.int64)
    ood_flag = torch.ones(len(ood_subset), dtype=torch.int64)
    
    
    merged_data = ConcatDataset([id_data, ood_subset])
    flags = torch.cat([id_flag, ood_flag])
    
    class MergedDataset(torch.utils.data.Dataset):
        def __init__(self, data, flags):
            self.data = data
            self.flags = flags
        
        def __getitem__(self, index):
            sample, flag = self.data[index], self.flags[index]
            return sample, flag
        
        def __len__(self):
            return len(self.data)
    
    
    return MergedDataset(merged_data, flags)
def generate_balanced_data(id_data, ood_data,ood_rate=0.5):
    """
    Merge two PyTorch datasets with a 1:1 ratio.

    Args:
        id_data (torch.utils.data.Dataset): PyTorch dataset containing in-distribution samples.
        ood_data (torch.utils.data.Dataset): PyTorch dataset containing out-of-distribution samples.

    Returns:
        torch.utils.data.Dataset: Merged dataset with a flag indicating the origin of each sample.
    """
    if  (ood_rate > 1):
        raise ValueError("ood_rate must be between 0 and 1")
    id_rate = 1-ood_rate

    if(ood_rate==0):
        sum_simple = len(id_data)
    else:
        sum_simple = min(len(id_data)/id_rate,len(ood_data)/ood_rate)        
    
    id_sample_count = int(sum_simple * id_rate)
    ood_sample_count = int(sum_simple * ood_rate)
    
    
    if len(id_data) > id_sample_count:
        id_subset_indices = random.sample(range(len(id_data)), id_sample_count)
        id_subset = Subset(id_data, id_subset_indices)
    else:
        id_subset = id_data
    
    
    if len(ood_data) > ood_sample_count:
        ood_subset_indices = random.sample(range(len(ood_data)), ood_sample_count)
        ood_subset = Subset(ood_data, ood_subset_indices)
    else:
        ood_subset = ood_data
    
    
    id_flag = torch.zeros(len(id_subset), dtype=torch.int64)
    ood_flag = torch.ones(len(ood_subset), dtype=torch.int64)
    
    
    merged_data = ConcatDataset([id_subset, ood_subset])
    flags = torch.cat([id_flag, ood_flag])
    
    
    class MergedDataset(Dataset):
        def __init__(self, data, flags):
            self.data = data
            self.flags = flags
        
        def __getitem__(self, index):
            sample, flag = self.data[index], self.flags[index]
            return sample, flag
        
        def __len__(self):
            return len(self.data)
    
    
    return MergedDataset(merged_data, flags)

def merge_datasets(dataset_list):
    """
    Merge a list of PyTorch datasets into a single dataset with equal sampling from each dataset.

    Args:
        dataset_list (list): A list of PyTorch datasets.

    Returns:
        ConcatDataset: A concatenated dataset.
    """
    
    min_num_samples = min(len(dataset) for dataset in dataset_list)

    
    subset_list = [Subset(dataset, torch.randperm(len(dataset))[:min_num_samples]) for dataset in dataset_list]

    
    merged_dataset = ConcatDataset(subset_list)

    return merged_dataset