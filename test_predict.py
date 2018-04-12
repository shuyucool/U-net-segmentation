import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from data import *
from unet import *

# mydata = dataProcess(512,512)
imgs_test = np.load('data/npydata/imgs_test.npy')
# imgs_test = mydata.load_test_data()
# myunet = myUnet()
#
# model = myunet.get_unet()
#
# model.load_weights('unet_first_train.hdf5')
#
# imgs_mask_test = model.predict(imgs_test, verbose=1)
#
# np.save('imgs_mask_test.npy', imgs_mask_test)

imgs_test_predict = np.load('imgs_mask_test_1.npy')
print(imgs_test.shape, imgs_test_predict.shape)

j = p = 0
n = 2
# plt.figure(figsize=(20, 4))
for i in range(0, 2):
    if i % 2 == 0 and i!=20:
        plt.figure(j)
        plt.gray()
        j += 1
    if p == 2:
        p = 0
    # plt.gray()
    ax = plt.subplot(2, n, (p-0)+1)
    plt.imshow(imgs_test[i].reshape(512, 512))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, (p - 0) + n + 1)
    plt.imshow(imgs_test_predict[i].reshape(512, 512))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    p += 1
    # plt.savefig(str(j)+'.png',dpi = 100)
plt.show()
