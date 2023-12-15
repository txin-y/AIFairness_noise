import torch
from torch import nn

class NModule(nn.Module):
    
    def __init__(self):
        super(NModule, self).__init__()
        
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def set_noise(self, dev_var_std):
        scale = self.op.weight.abs().max().item()
        self.noise = torch.randn_like(self.op.weight) * scale * dev_var_std
        
    def is_cuda(self):
        return next(self.parameters()).is_cuda()
    
class NConv2d(NModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.clear_noise()
        if self.op.bias is not None:
            nn.init.zeros_(self.op.bias)
        nn.init.kaiming_normal_(self.op.weight)
    
    def forward(self, x):
        
        return nn.functional.conv2d(x, self.op.weight + self.noise, self.op.bias, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)

class NLinear(NModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.clear_noise()
        if self.op.bias is not None:
            nn.init.zeros_(self.op.bias)
        nn.init.kaiming_normal_(self.op.weight)
    
    def forward(self, x):

        return nn.functional.linear(x, self.op.weight + self.noise, self.op.bias)

class NModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def set_noise(self, dev_var_std):
        for mo in self.modules():
            if isinstance(mo, NModule):
                mo.set_noise(dev_var_std)
    
    def clear_noise(self):
        for mo in self.modules():
            if isinstance(mo, NModule):
                mo.clear_noise()
    
    def unpack_flattern(self, x):
        # print(x.size())
        return x.view(-1, self.num_flat_features(x))

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



