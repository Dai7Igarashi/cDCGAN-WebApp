# coding: UTF-8

import argparse
import time
import os
import numpy as np
from PIL import Image
from io import BytesIO
import base64

import chainer

from conditionalDCGAN import Generator

def generate_image(label_array, gpu):

    print('=== Generate Image ===')

    # 学習済みGenerator設定
    n_hidden = 100
    gen = Generator(n_hidden=n_hidden)
    chainer.serializers.load_npz('generator.npz', gen)

    # cpu or gpu
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        gen.to_gpu()
    xp = chainer.cuda.cupy if gpu >= 0 else np

    # 入力ノイズデータ作成(バッチサイズ2)
    z = chainer.Variable(xp.asarray(gen.make_hidden(2)))

    # ラベル整形 str -> np.array
    label = xp.array(label_array, dtype=xp.float32)

    label = xp.vstack((label, label)).reshape(2, 10, 1, 1)

    # 画像生成
    with chainer.using_config('train', False):
        x_fake = gen(z, label)

    x_fake = chainer.cuda.to_cpu(x_fake.data)
    x_fake = np.asarray(np.clip(x_fake * 255, 0.0, 255.0), dtype=np.uint8)

    _, _, H, W = x_fake.shape
    x = x_fake[0].reshape(H, W)

    # Image.fromarray(x).save('static/images/mnist.png')
    image = Image.fromarray(x)
    buffer = BytesIO()
    image.save(buffer, format='png')
    encode = base64.b64encode(buffer.getvalue())

    return encode