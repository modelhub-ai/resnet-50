from modelhublib.processor import ImageProcessorBase
import PIL
import SimpleITK
import numpy as np
import json


class ImageProcessor(ImageProcessorBase):

    def _preprocessBeforeConversionToNumpy(self, image):
        if isinstance(image, PIL.Image.Image):
            image = image.resize((224,224), resample = PIL.Image.LANCZOS)
        elif isinstance(image, SimpleITK.Image):
            newSize = [224, 224]
            referenceImage = SimpleITK.Image(newSize, image.GetPixelIDValue())
            referenceImage.SetOrigin(image.GetOrigin())
            referenceImage.SetDirection(image.GetDirection())
            referenceImage.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(newSize,
                                                                        image.GetSize(),
                                                                        image.GetSpacing())])
            image = SimpleITK.Resample(image, referenceImage)
        else:
            raise IOError("Image Type not supported for preprocessing.")
        return image

    def _preprocessAfterConversionToNumpy(self, npArr):
        if npArr.shape[1] > 3:
            npArr = npArr[:,0:3,:,:]
        elif npArr.shape[1] < 3:
            npArr = npArr[:,[0],:,:]
            npArr = np.concatenate((npArr, npArr[:,[0],:,:]), axis = 1)
            npArr = np.concatenate((npArr, npArr[:,[0],:,:]), axis = 1)
        npArr = npArr.reshape(3, 224, 224)
        npArr = self._preprocess(npArr)
        npArr = npArr.reshape(1, 3, 224, 224)
        return npArr

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

    def _preprocess(self, img_data):
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[0]):
             # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
            norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        return norm_img_data
