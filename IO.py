import tarfile, pickle, cv2, re
import numpy as np
from IPython import embed

def load_flowers17(path, shape):
    tar = tarfile.open(path)    #   read 17flowers.tar.gz
    names = sorted(tar.getnames())  #   get names of files
    dataset = [[] for i in range(17)]
    for name in names:              
        if name[-4:] == '.jpg': #   if suffix is 'jpg'
            data = tar.extractfile(name).read() #   read from file

            #   decode into image
            img = cv2.imdecode(\
                    np.fromstring(data, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, tuple(shape[:2]))
            #   get image index
            index = int(re.compile('image_(.*?)\.jpg').findall(name)[0])
            #   calculate label
            label = (index - 1) // 80
            dataset[label].append(img)
    return dataset
            
        

def load_cifar_10(path):
    tar = tarfile.open(path)
    names = sorted(tar.getnames())
    train, test = {'image': [], 'label': []}, {'image': [], 'label': []}

    for name in names:
        if 'data_batch' in name:
            data = pickle.loads(tar.extractfile(name).read(),\
                    encoding='bytes')
            train['image'].append(data[b'data'].reshape((-1, 32, 32, 3)))
            train['label'] += data[b'labels']
        if 'test' in name:
            data = pickle.loads(tar.extractfile(name).read(),\
                    encoding='bytes')
            test['image'].append(data[b'data'].reshape((-1, 32, 32, 3)))
            test['label'] += data[b'labels']

    images = np.array(train['image'])
    train['image'] = images.reshape((-1, *images.shape[2:]))
    return train, test
            
if __name__ == '__main__':
    # train, test = load_cifar_10('./data/cifar-10-python.tar.gz')
    load_flower17('./data/17flowers.tgz')
