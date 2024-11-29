import torch

path = 'checkpoint/chapter_title_gen/pegasus_batch_16/checkpoint.pth'

ckpt = torch.load(path)

print("Checkpoint keys:", ckpt.keys())
print(f"epoch : {ckpt['epoch']}")
print(f"best_result : {ckpt['best_result']}")