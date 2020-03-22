#pytorch version 1.4.0 checked

import torch

with torch.no_grad():
    model = torch.load('./ckpt/NME(4.46).pth', map_location='cpu').eval()

example = torch.randn(1,3,256,256)

traced_script_module = torch.jit.trace(model, example)
output = traced_script_module(torch.ones(1, 3, 256, 256))
print(output)

traced_script_module.save("./ckpt/hg4_mobile.pt")

print("model saved")
