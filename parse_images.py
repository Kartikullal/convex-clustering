import struct
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class parse_images():
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path

    
    def parse_images(self):
        images = []
        with open(self.image_path, 'rb') as fp:
            header = struct.unpack('>4i', fp.read(16))
            magic, size, width, height = header

            if magic != 2051:
                raise RuntimeError("'%s' is not an MNIST image set." % self.image_path)

            chunk = width * height
            for _ in range(size):
                img = struct.unpack('>%dB' % chunk, fp.read(chunk))
                img_np = np.array(img, np.uint8)
                images.append(img_np)

        return np.array(images).astype(int)

    
    def parse_labels(self):
        with open(self.label_path, 'rb') as fp:
            header = struct.unpack('>2i', fp.read(8))
            magic, size = header

            if magic != 2049:
                raise RuntimeError("'%s' is not an MNIST label set." % self.label_path)

            labels = struct.unpack('>%dB' % size, fp.read())

        return np.array(labels, np.int32)

    