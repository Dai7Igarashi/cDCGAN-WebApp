# coding: UTF-8

import base64
from flask import Flask, request, render_template
import os

from test import generate_image


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chainer', methods=['POST'])
def generate_mnist():
    ranges = request.form.getlist('range[]')
    # print('#arr: {}'.format(ranges))

    # conditionalDCGAN
    encode = generate_image(ranges)

    return encode

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)