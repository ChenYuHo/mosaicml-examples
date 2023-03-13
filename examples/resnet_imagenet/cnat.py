import torch
import torch.distributed as dist
import cupy
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import random

def cnat(tensor):
    cupy_tensor = cupy.fromDlpack(to_dlpack(tensor))
    tensor_cast = cupy.abs(cupy_tensor).view(cupy.int32)
    sign = cupy.sign(cupy_tensor)
    exp = tensor_cast & cupy.int32(0b01111111100000000000000000000000)
    mantissa = tensor_cast & cupy.int32(0b00000000011111111111111111111111)
    exp_add_one = mantissa > cupy.random.randint(low=0, high=0b00000000011111111111111111111111,
                                                 size=cupy_tensor.shape, dtype=cupy.int32)
    exponent = cupy.where(exp_add_one, exp + 0b00000000100000000000000000000000, exp).view(cupy.float32)
    tensor_compressed = cupy.multiply(exponent, sign)
    return from_dlpack(tensor_compressed.toDlpack())

def decompress(tensor, ctx):
    """Simulates two-way compression"""
    cupy_tensor = cupy.fromDlpack(to_dlpack(tensor))
    tensor_cast = cupy_tensor.view(cupy.int32)
    sign = cupy.sign(cupy_tensor)
    exp = tensor_cast & cupy.int32(0b01111111100000000000000000000000)
    mantissa = tensor_cast & cupy.int32(0b00000000011111111111111111111111)
    cupy.random.seed(exp.ravel()[0].item())
    exp_add_one = mantissa > cupy.random.randint(low=0, high=0b00000000011111111111111111111111,
                                                 size=cupy_tensor.shape,
                                                 dtype=cupy.int32)
    exponent = cupy.where(exp_add_one, exp + 0b00000000100000000000000000000000, exp).view(cupy.float32)
    tensor_decompressed = cupy.multiply(exponent, sign)
    return from_dlpack(tensor_decompressed.toDlpack())

def cnat_compress_hook(process_group, bucket):
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    compressed_tensor = cnat(bucket.buffer()).div_(world_size)

    return (
        dist.all_reduce(compressed_tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def cnat_random_block(grad, block_size=262144):
    numel = grad.numel()
    for i in range(0, numel, block_size):
        if random.random() < 0.4:
            grad[i:i+block_size] = cnat(grad[i:i+block_size])
    return grad

def block_cnat_compress_hook(process_group, bucket):
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    compressed_tensor = cnat_random_block(bucket.buffer()).div_(world_size)

    return (
        dist.all_reduce(compressed_tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )


class CNat(Algorithm):
    def __init__(self, deterministic=False, return_float=False):
        self.deterministic = deterministic
        self.return_float = return_float
        self.p = 0

    def match(self, event, state):
        return event == Event.AFTER_TRAIN_BATCH

    def apply(self, event, state, logger):
        if self.p < 1:
            for param in state.model.parameters():
                print(param.grad)
            self.p += 1
