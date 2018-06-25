# coding: UTF-8

import numpy as np
from PIL import Image
from io import BytesIO
import base64

import chainer

from conditionalDCGAN import Generator

def generate_image(label_array):

    # print('=== Generate Image ===')

    # 学習済みGenerator設定
    n_hidden = 100
    gen = Generator(n_hidden=n_hidden)
    chainer.serializers.load_npz('generator.npz', gen)

    # 入力ノイズデータ作成(バッチサイズ2)
    z = chainer.Variable(np.asarray(gen.make_hidden(2)))

    # ラベル整形 str -> np.array
    label = np.array(label_array, dtype=np.float32)

    label = np.vstack((label, label)).reshape(2, 10, 1, 1)

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