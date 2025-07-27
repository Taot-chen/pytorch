#!/usr/bin/env python

# Run with: torchrun --nnodes=1 --nproc_per_node=1 ddp.py

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch._dynamo as dynamo

def run_epoch(model, loss_fn, optimizer, inputs, labels):
    for i in range(3):
        print(f">>> Iteration {i}")
        outputs = model(inputs)
        loss_fn(outputs, labels).backward()
        optimizer.step()

def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    efficientnet = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_efficientnet_b0",
        pretrained=False,
    )
    
    model = efficientnet.to(device_id, memory_format = torch.channels_last)
    model = DDP(model, device_ids = [device_id], bucket_cap_mb=4)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    optimizer.zero_grad()
    
    
    inputs = torch.randn((4, 3, 224, 224), device="cuda")
    inputs = inputs.to(memory_format=torch.channels_last)
    labels = torch.randn(4, 1000).to(device_id)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> Parameters: {num_params}")
    
    ret = dynamo.explain(
        model, inputs
    )
    
    print(f"{len(ret.graphs)} graphs")
    
    model = torch.compile(model, backend="eager")
    run_epoch(model, loss_fn, optimizer, inputs, labels)

if __name__ == "__main__":
    demo_basic()
