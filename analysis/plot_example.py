import torch
from networks.dcgan import Generator
from matplotlib import pyplot as plt

h = 32
w = 32
dim = 128
generator = Generator(dim, h, w, condition_vector_size=2)

model_file = '/Users/haihabi/Downloads/model_best.pt'
generator.load_state_dict(torch.load(model_file, map_location='cpu'))
generator.eval()


# rain_rate


def sample_rain_field(rain_rate, n_peaks):
    with torch.no_grad():
        cond = torch.Tensor([rain_rate, n_peaks]).reshape([1, -1])
        z = torch.randn([1, 128])
        return generator(z, cond).cpu().numpy()[0, 0, :, :]


for i, pixel_rate in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
    rain_field = sample_rain_field(pixel_rate, 6)
    plt.subplot(2, 5, i + 1)
    plt.imshow(rain_field)

for i, n_peaks in enumerate([3, 6, 15, 20, 30]):
    rain_field = sample_rain_field(0.3, n_peaks)
    plt.subplot(2, 5, i + 6)
    plt.imshow(rain_field)
plt.show()
