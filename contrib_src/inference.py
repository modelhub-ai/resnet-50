import json
from processing import ImageProcessor
from modelhublib.model import ModelBase
import mxnet as mx
import numpy as np
from collections import namedtuple
from mxnet.contrib.onnx.onnx2mx.import_model import import_model


class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # get context - cpu
        ctx = mx.cpu()
        sym, arg_params, aux_params = import_model('model/model.onnx')
        # Load module
        mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],
                 label_shapes=mod._label_shapes)
        mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

        self._model = mod

    def infer(self, input):
        # load preprocessed input
        inputAsNpArr = self._imageProcessor.loadAndPreprocess(input)
        # Run inference with mxnet
        batch = namedtuple('Batch', ['data'])
        self._model.forward(batch([inputAsNpArr]))
        # Take softmax to generate probabilities
        scores = mx.ndarray.softmax(self._model.get_outputs()[0][0]).asnumpy()
        # postprocess results into output
        output = self._imageProcessor.computeOutput(scores)
        return output
