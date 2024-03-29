import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result


class attention2d(nn.Module):
    def __init__(self,in_planes,ratios,K,temperature,init_weight = True):
        super(attention2d,self).__init__()
        assert temperature % 3 == 1 # for reducing τ temperature from 30 to 1 linearly in the first 10 epochs.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        
        self.fc1   = nn.Conv2d(in_planes,hidden_planes,1,bias = False)
        # self.relu  = nn.ReLU()
        self.fc2   = nn.Conv2d(hidden_planes,K,1,bias = True)
        self.temperature = temperature
        
        if init_weight:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    
    def update__temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
    
    def forward(self,z):
        z = self.avgpool(z)
        z = self.fc1(z)
        # z = self.relu(z)
        z = F.relu(z)
        z = self.fc2(z)
        z = z.view(z.size(0),-1) 
        # z = self.fc2(z).view(z.size(0), -1)
          
        return F.softmax(z/self.temperature,1) 
    
class Dynamic_conv2d(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,ratio = 0.25,stride = 1,padding = 0,dilation = 1,groups = 1,bias = True,K = 4,temperature = 34,init_weight = True):
        super(Dynamic_conv2d,self).__init__()
        
        if in_planes%groups != 0:
            raise ValueError('Error : in_planes%groups != 0')
        self.in_planes    = in_planes
        self.out_planes   = out_planes
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.groups       = groups
        self.bias         = bias
        self.K            = K
        self.attention    = attention2d(in_planes,ratio,K,temperature)
        self.weight       = nn.Parameter(torch.randn(K,out_planes,in_planes//groups,kernel_size,kernel_size),requires_grad = True)
        
        if bias :
            self.bias = nn.Parameter(torch.Tensor(K,out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])
    def update_temperature(self):
        self.attention.update__temperature()
    
    def forward(self,z):
        
#         Regard batch as a dimensional variable, perform group convolution,
#         because the weight of group convolution is different, 
#         and the weight of dynamic convolution is also different
        softmax_attention = self.attention(z)
        batch_size ,in_planes,height,width = z.size()
        z = z.view(1,-1,height,width) # changing into dimension for group convolution
        weight = self.weight.view(self.K,-1)
        
#         The generation of the weight of dynamic convolution,
#         which generates batch_size convolution parameters 
#         (each parameter is different) 
        aggregate_weight = torch.mm(softmax_attention,weight).view(-1,self.in_planes,self.kernel_size,self.kernel_size)# expects two matrices (2D tensors)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention,self.bias).view(-1)
            output = F.conv2d(x,weight = aggregate_weight,bias = aggregate_bias,stride = self.stride,padding = self.padding,
                             dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(z,weight = aggregate_weight,bias = None,stride = self.stride,padding = self.padding,
                             dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2),output.size(-1))
        # print('2d-att-for')
        return output        

if __name__ == '__main__':
    x = torch.randn(24, 3,  20)