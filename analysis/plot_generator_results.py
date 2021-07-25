import matplotlib.pyplot as plt
from networks.factory import get_rain_gan_function

if __name__ == '__main__':
    h, w = 32, 32
    rain_field_function = get_rain_gan_function(h, w)
    k = 4
    sample = rain_field_function(rain_coverage=0.3,
                                 n_peaks=5,
                                 peak_rain_rate=1,
                                 batch_size=k ** 2)
    for i in range(k):
        for j in range(k):
            plt.subplot(k, k, i + 1 + 4 * j)
            plt.imshow(sample[i + 4 * j, :, :])
    plt.show()
    sample_rain_field = get_rain_gan_function(32, 32)
    ###########################
    # Model Collapse Plot
    ###########################
    k = 4
    sample = sample_rain_field(0.3, 5, 1, batch_size=k ** 2)
    for i in range(k):
        for j in range(k):
            plt.subplot(k, k, i + 1 + 4 * j)
            plt.imshow(sample[i + 4 * j, :, :])
    plt.show()
    ###########################
    # Rain Coverage Plot
    ###########################
    rain_coverage = [0.1, 0.4, 0.7, 0.9]
    for i, r in enumerate(rain_coverage):
        plt.subplot(1, len(rain_coverage), i + 1)
        sample = sample_rain_field(r, 5, 1, batch_size=1)
        plt.imshow(sample[0, :, :])
        plt.title(f"$R_c={r}$")
    plt.show()
    ###########################
    # N Peaks Plot
    ###########################
    n_peaks_array = [5, 10, 20, 30]
    for i, n_p in enumerate(n_peaks_array):
        plt.subplot(1, len(n_peaks_array), i + 1)
        sample = sample_rain_field(0.3, n_p, 1, batch_size=1)
        plt.imshow(sample[0, :, :])
        plt.title(f"$N_p={n_p}$")
    plt.show()
