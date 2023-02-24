import torch
import torch.nn as nn
in_planes = 256

init_U = torch.randn(in_planes, in_planes) * 0.1
init_U = init_U / torch.norm(init_U, dim=1, keepdim=True)
init_b = torch.randn(in_planes)
U = nn.Parameter(init_U)
# bias = nn.Parameter(init_b)

# -----Implementation of Convolution layer
conv1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
device = conv1.weight.data.device
U_conv = init_U[:, :, None, None]
conv1.weight.data = U_conv.to(device)
# conv1.bias.data = init_b.to(device)

x = torch.rand([1, in_planes, 100, 100])
B, C, H, W = x.shape

x2 = x.reshape(B, C, H * W)
U2 = U[None, ...].expand(B, C, C)
# b2 = bias[None, ...].expand(B, C)[..., None]
atte_map = torch.matmul(U2, x2)# + b2
atte_map = atte_map.reshape(B, C, H, W)

atte_map2 = conv1(x)
# print(torch.abs(atte_map - atte_map2)[0, 0, ...])
print(torch.sum(torch.abs(atte_map - atte_map2)))
# loss1 = torch.sum(atte_map)
# loss2 = torch.sum(atte_map2)
#
# loss1.backward()
# loss2.backward()
# grad = U.grad
# print(torch.mean(grad), torch.min(grad), torch.max(grad))
# grad = conv1.weight.grad
# print(torch.mean(grad), torch.min(grad), torch.max(grad))

U_get = conv1.weight[:, :, 0, 0]
utu = torch.matmul(U_get, U_get.transpose(1, 0))
target = torch.eye(len(U_get), device=x.device)
og_loss = torch.dist(utu, target, p=2)

utu = torch.matmul(U, U.transpose(1, 0))
target = torch.eye(len(U), device=x.device)
F2norm = torch.dist(utu, target, p=2)

print(og_loss, F2norm)
og_loss.backward()
F2norm.backward()

grad = U.grad
print(torch.mean(grad), torch.min(grad), torch.max(grad))
grad = conv1.weight.grad
print(torch.mean(grad), torch.min(grad), torch.max(grad))

# x = torch.Tensor([[[1, 2],
#                   [3, 4]],
#                   [[-1, -2],
#                    [-3, -4]]]).float()
# x = x[None, ...]
# U = torch.Tensor([[1, -1],
#                   [1, -1]]).float()
# U2 = U[None, ...].expand(1, 2, 2)
# x2 = x.reshape(1, 2, 2 * 2)
# atte_map = torch.matmul(U2, x2)
# atte_map = atte_map.reshape(1, 2, 2, 2)
#
# conv1 = nn.Conv2d(2, 2, 1, bias=False)
# conv1.weight.data = U[:, :, None, None]
# atte_map2 = conv1(x)
#
# print(atte_map)
# print(atte_map2)
