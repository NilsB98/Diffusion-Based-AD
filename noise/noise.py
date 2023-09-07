import torch
import math


def rand_perlin_2d(shape: tuple, res: tuple, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
            torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                        dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def perlin_2d_batch(shape: tuple, res: tuple, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    # TODO try random noise in all channels instead of repeat
    img_noise = lambda: rand_perlin_2d_octaves(shape[-2:], res, octaves=5).unsqueeze(0).repeat(3, 1, 1)
    return torch.stack([img_noise() for _ in range(shape[0])])


def rand_perlin_2d_octaves(shape: tuple, res: tuple, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    noise = rand_perlin_2d((256, 256), (8, 8))
    plt.figure()
    plt.imshow(noise, cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.savefig('perlin.png')
    plt.close()

    batch_noise = perlin_2d_batch((8, 3, 128, 128), (8, 8))
    noise = rand_perlin_2d_octaves((128, 128), (8, 8), 5)
    plt.figure()
    plt.imshow(noise, cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.savefig('perlino.png')
    plt.close()

    noise = torch.randn((256, 256))
    plt.figure()
    plt.imshow(noise, cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.savefig('gaussian.png')
    plt.close()
