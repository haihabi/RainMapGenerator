import torch
from networks.dcgan import Generator
from matplotlib import pyplot as plt

h = 32
w = 32
dim = 128
z_size = 128

g = Generator(dim, h, w)

g.load_state_dict(torch.load("C:\\model_best.pt", map_location='cpu'))
z_0 = torch.randn([1, z_size])

out = g(z_0)
#####################################
# preper
#####################################
u, s, v = torch.svd(g.preprocess[0].weight)
v_ref = v[0, :].unsqueeze(dim=-1)
v_target = v[0, :].unsqueeze(dim=-1)
# print(v_ref.shape, v_target.shape)
I = torch.eye(z_size)
p_target = torch.matmul(v_target, v_target.transpose(0, 1))
p_ref = torch.matmul(v_ref, v_ref.transpose(0, 1))
p_target_proj = I - p_target
p_ref_proj = I - p_ref

##########################
#
##########################
n_steps = 10
delta = torch.tensor([2 * 3.14 / n_steps])
# n = 2
# b = torch.matmul(z_0, p_target_proj.transpose(0, 1))
p_z = torch.matmul(z_0, p_target_proj.transpose(0, 1))
norm = torch.sqrt(torch.pow(p_z, 2).sum(dim=-1))
norm_z = torch.sqrt(torch.pow(z_0, 2).sum(dim=-1))
theta = torch.arccos(torch.matmul(z_0, p_ref) / norm_z) * torch.sign(torch.sum(p_z * v, dim=-1))
u = p_z / norm
for n in range(n_steps):
    plt.subplot(1, n_steps, n + 1)
    z_n = norm_z * (u * torch.cos(n * delta + theta) + v_target.transpose(0, 1) * torch.sin(
        n * delta + theta))
    out_n = g(z_n)
    plt.imshow(out_n.detach().numpy()[0, 0, :, :])
plt.show()
# print(z_n.shape,norm.shape,b.shape)
# print("a")
