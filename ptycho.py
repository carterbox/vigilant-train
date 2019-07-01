import os
import importlib

import click
import tike.ptycho
import numpy as np
import matplotlib.pyplot as plt
import xdesign


def bad_probe(probe, threshold=10.0):
    """Apply a low-pass filter to the probe in frequency space."""

    def center_range(L):
        return np.arange(L) - 0.5 * (L - 1)

    # Generate the low-pass filter
    x, y = np.meshgrid(
        center_range(probe.shape[-2]),
        center_range(probe.shape[-1]),
    )
    mask = x * x + y * y > threshold * threshold * probe.shape[
        -2] * probe.shape[-1] * 0.25
    # Apply the filer in frequency space
    # data = np.fft.fft2(probe, axes=(-2, -1))
    data = probe
    data = np.fft.fftshift(data, axes=(-2, -1))
    data[..., mask] = 0
    data = np.fft.ifftshift(data, axes=(-2, -1))
    # new_probe = np.fft.ifft2(data, axes=(-2, -1))
    return data


@click.command()
@click.argument('folder')
@click.option('--size', default=256, help='The width of the object.')
@click.option('--probe-width', default=113, help='The width of the probe.')
@click.option('--probe-shape', default="gaussian")
@click.option('--shift',
              default=8,
              type=float,
              help='The distance between positions in pixels.')
@click.option('--threshold', default=-1, type=float)
@click.option(
    '--detector',
    default=3.0,
    type=float,
    help='The width of the detector as a multiple of the probe width.')
@click.option('--show', is_flag=True)
@click.option('--noise', is_flag=True)
@click.option('--pad', is_flag=True)
@click.option('--flux', default=1.0, help='Integral of the probe intensity.')
def main(folder, size, probe_width, shift, threshold, detector, show, noise,
         flux, probe_shape, pad):
    """Simulate ptychography experiment with boxcar probe."""
    # Store the input params in a log file.
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "log.txt"), 'w') as log:
        for key in list(locals().keys()):
            value = eval(key)
            print("'{}': {}".format(key, value), file=log)
    # ### Load the images
    amplitude = plt.imread(
        os.path.expanduser(
            "./images/Cryptomeria_japonica-{:04d}.tif".format(size))) / 255
    phase = plt.imread(
        os.path.expanduser("./images/Erdhummel_Bombus_terrestris-{:04d}.tif".
                           format(size))) / 255 * np.pi
    original = amplitude * np.exp(1j * phase)
    if show:
        tike.plot_phase(original)

    if pad is True:
        original = np.pad(original, (probe_width, probe_width + int(shift)),
                          mode='edge')

    # ### Define the trajectory
    pw = probe_width  # probe width
    v, h = np.meshgrid(
        np.arange(0, original.shape[0] - pw, shift),
        np.arange(0, original.shape[0] - pw, shift),
        indexing='ij',
    )

    # ### Define the probe
    if probe_shape == 'gaussian':
        weights = tike.ptycho.gaussian(pw)
        pamp = tike.constants.sum_square_norm(weights, flux)
        probe = pamp * np.exp(1j * weights * 0.2)
    elif probe_shape == 'mura':
        weights = xdesign.mura_2d(pw).astype(np.float32)
        weights[weights == 0] = 0.2  # mask is not completely blocking
        pamp = tike.constants.sum_square_norm(weights, flux)
        probe = pamp * np.exp(1j * weights * 0.2)
    elif probe_shape == 'gaussian-random':
        np.random.seed(0)
        weights = tike.ptycho.gaussian(pw)
        pamp = tike.constants.sum_square_norm(weights, flux)
        probe = pamp * np.exp(1j * weights * np.pi * 0.5 *
                              (np.random.rand(pw, pw) * 2 - 1))
    elif probe_shape == 'deBrujin':
        weights = plt.imread('./thumbnail_output-4x4.png')[256:256 +
                                                           pw, 256:256 + pw]
        pamp = tike.constants.sum_square_norm(weights, flux)
        probe = pamp * np.exp(1j * weights * 0.2)
    else:
        raise ValueError("Unknown probe shape.")

    # probe = bad_probe(probe, threshold)

    if show:
        tike.plot_phase(probe)

    # ## Simulate data acquisition

    # Then what we see at the detector is the wave propagation
    # of the near field wavefront
    data = tike.ptycho.simulate(
        data_shape=np.ones(2, dtype=int) * int(pw * detector),
        probe=probe,
        v=v,
        h=h,
        psi=original,
    )
    # data = bad_probe(data, threshold)
    if noise:
        np.random.seed(0)
        data = np.random.poisson(data)
    else:
        data[data < threshold] = 0
    if show:
        plt.figure()
        plt.imshow(np.fft.fftshift(np.log10(data[len(data) // 2])))
        plt.colorbar()
        # plt.figure()
        # plt.imshow(np.fft.fftshift(np.log10(data[11])))
        # plt.colorbar()
        plt.show()

    filename = os.path.join(folder, 'data')
    np.save(filename, np.sum(data, axis=0))
    plt.figure()
    plt.imshow(np.fft.fftshift(np.log10(data[len(data) // 2])))
    plt.colorbar()
    plt.savefig(filename)
    filename = os.path.join(folder, 'mdata')
    np.save(filename, np.sum(data, axis=0))
    plt.figure()
    plt.imshow(np.fft.fftshift(np.log10(np.mean(data, axis=0))))
    plt.colorbar()
    plt.savefig(filename)

    # # Reconstruct
    #
    # Now we need to try and reconstruct psi.

    # Start with a guess of all zeros for psi
    new_psi = np.ones(original.shape, dtype=complex)
    for i in range(50):
        new_psi = tike.ptycho.reconstruct(
            data=data,
            probe=probe,
            v=v,
            h=h,
            psi=new_psi,
            algorithm='cgrad',
            num_iter=10,
            gamma=1,
        )
        if show:
            tike.plot_phase(new_psi)
        filename = os.path.join(folder, 'psi-{:03d}'.format((i + 1) * 10))
        np.save(filename, new_psi)
        plt.imsave(filename + '-amp.jpg', np.abs(new_psi), vmin=0, vmax=1)
        plt.imsave(
            filename + '-phz.jpg',
            np.angle(new_psi),
            vmin=-np.pi,
            vmax=np.pi,
            cmap=plt.cm.twilight,
        )


if __name__ == '__main__':
    main()
