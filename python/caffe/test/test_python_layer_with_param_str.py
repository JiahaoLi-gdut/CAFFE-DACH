import unittest
import tempfile
import os
import six

import caffe


class SimpleParamLayer(caffe.Layer):
    """A layer that just multiplies by the numeric value of its param string"""

    def setup(self, bottom, top):
        try:
            self.value = float(self.param_str)
        except ValueError:
            raise ValueError("Parameter string must be a legible float")

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top, preforward_flag):
        top[0].data[...] = self.value * bottom[0].data

    def backward(self, top, propagate_down, bottom, prebackward_flag):
        bottom[0].diff[...] = self.value * top[0].diff


def python_param_net_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        input: 'data' input_shape { dim: 10 dim: 9 dim: 8 }
        layer { type: 'Python' name: 'mul10' bottom: 'data' top: 'mul10'
          python_param { module: 'test_python_layer_with_param_str'
                layer: 'SimpleParamLayer' param_str: '10' } }
        layer { type: 'Python' name: 'mul2' bottom: 'mul10' top: 'mul2'
          python_param { module: 'test_python_layer_with_param_str'
                layer: 'SimpleParamLayer' param_str: '2' } }""")
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(),
    'Caffe built without Python layer support')
class TestLayerWithParam(unittest.TestCase):
    def setUp(self):
        net_file = python_param_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self, preforward_flag):
        x = 8
        self.net.blobs['data'].data[...] = x
        self.net.forward(preforward_flag)
        for y in self.net.blobs['mul2'].data.flat:
            self.assertEqual(y, 2 * 10 * x)

    def test_backward(self, prebackward_flag):
        x = 7
        self.net.blobs['mul2'].diff[...] = x
        self.net.backward(prebackward_flag)
        for y in self.net.blobs['data'].diff.flat:
            self.assertEqual(y, 2 * 10 * x)