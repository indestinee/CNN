import cv2, data_provider
import numpy as np
dp = data_provider.dp

data = dp.val

for i in range(data.image.shape[0]):
    cv2.imshow('%d' % np.argmax(data.label[i]), data.image[i])
    cv2.waitKey(100)
