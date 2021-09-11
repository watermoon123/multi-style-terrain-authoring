from psychopy.visual import filters
import numpy as np

def butter2d(image, cutoff, n):
    avr = np.average(image)
    std = np.std(image)
    image = (image - avr) / std
    img_freq = np.fft.fft2(image)
    lp_filt = filters.butter2d_lp(size=image.shape, cutoff=cutoff, n=n)
    img_filt = np.fft.fftshift(img_freq) * lp_filt
    s = np.real(np.fft.ifft2(np.fft.ifftshift(img_filt)))
    # s = np.clip(s, a_min=-1.0, a_max=1.0)
    return (s * std) + avr