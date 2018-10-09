import sys
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

def get_batch(data, batch_size, allow_smaller_batch=False):
    N = data.shape[0]
    indexes = torch.randperm(N)
    for i in range(0, N, batch_size):
        if not allow_smaller_batch and i + batch_size > N - 1:
            break
        yield data[indexes[i: i + batch_size]]


def visualize(net, data, in_size, dtype, seed=42):
    np.random.seed(seed=seed)
    in_size = tuple([-1] + in_size)

    idx = np.random.choice(range(0, data.shape[0]), size=10, replace=False)
    images = torch.from_numpy(data[idx]).view(in_size).type(dtype)
    images = Variable(images)
    reconstructions = net(images)
    print('code size = %s' % (str(net.encode(images).shape[1:]))) if 'encode' in dir(net) else None
    reconstructions = reconstructions.data.cpu().numpy()
    images = images.data.cpu().numpy()
    reconstructions = reconstructions.reshape((-1, 1, 28, 28))
    images = images.reshape((-1, 1, 28, 28))

    fig = plt.figure(figsize=(6, 6))
    for i in range(10):
        a = fig.add_subplot(10, 10, i + 1)
        b = fig.add_subplot(10, 10, i + 11)
        a.axis('off')
        b.axis('off')
        im = images[i - 1].squeeze()
        rec = reconstructions[i - 1].squeeze()
        a.imshow(im, cmap='Greys_r')
        b.imshow(rec, cmap='Greys_r');


def count_parameters(model):
    p1 = sum(p.numel() for p in model.parameters())
    p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{:,} parameters".format(p1))
    print("{:,} trainable parameters".format(p2))


def progress(loss, epoch, max_epochs, step, batch, N, start, net=None, data=None, in_size=None, seed=42):
    progress = min(step * batch / N * 100, 100)
    flush = '\n' if progress >= 100 else ''
    digits = len(str(max_epochs))
    elapsed = time.time() - start
    v = (progress / 100 * N) / elapsed
    progress = '\r[%*d/%*d][%50s] %2d%% – loss %.5f - %4is [%i fps]%s' %\
                (digits, epoch, digits, max_epochs, '=' * int(progress/2) + ' ' * (50 - int(progress/2)), progress, np.mean(loss), elapsed, v, flush)
    sys.stdout.write(progress)
