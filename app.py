# coding: UTF-8

import base64
from flask import Flask, request, render_template

from test import generate_image


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chainer', methods=['POST'])
def generate_mnist():
    ranges = request.form.getlist('range[]')
    print('#arr: {}'.format(ranges))

    # conditionalDCGAN
    generate_image(ranges, -1)

    image = open('./static/images/mnist.png', 'rb').read()
    encode = base64.b64encode(image)

    return encode

if __name__ == '__main__':
    app.run()