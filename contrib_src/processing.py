import numpy as np
import os
import mxnet as mx
from modelhublib.processor import ImageProcessorBase
import PIL
import SimpleITK
import numpy as np
import json
from mxnet.gluon.data.vision import transforms

class ImageProcessor(ImageProcessorBase):

    def _preprocessBeforeConversionToNumpy(self, image):
        if isinstance(image, PIL.Image.Image):
            image = np.array(image).astype(np.float32)
            if len(image.shape) > 2:
                image = image[:,:,0:3]
            else:
                image = np.stack((image,)*3, axis=-1)
            arr = mx.nd.array(image)
            transform_fn = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            arr = transform_fn(arr)
            arr = arr.expand_dims(axis=0)
            return arr.asnumpy()
        else:
            raise IOError("Image Type not supported for preprocessing.")

    def _preprocessAfterConversionToNumpy(self, npArr):
        return mx.nd.array(npArr)

    def computeOutput(self, inferenceResults):
        probs = np.squeeze(np.asarray(inferenceResults))
        with open("model/labels.json") as jsonFile:
            labels = json.load(jsonFile)
        result = []
        for i in range (len(probs)):
            obj = {'label': str(labels[str(i)]),
                    'probability': float(probs[i])}
            result.append(obj)
        return result
