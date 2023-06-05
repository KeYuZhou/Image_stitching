import cv2
import numpy as np
from DetectAndDescriptor import detectAndDescribe_sift
from Matcher import Flann_Match
import matplotlib.pyplot as plt

def warp(imageA,imageB,H):
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    # 将图片B传入result图片最左端
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    return result
class PanoramaStitcher:
    def __init__(self, imgs):
        self.imgs = imgs
        self.kpts = []
        self.descriptors = []
        self.matchs = []

    def run(self):
        for i in range(len(self.imgs)):
            kpt, des = detectAndDescribe_sift(self.imgs[i])
            self.kpts.append(kpt)
            self.descriptors.append(des)

        for i in range(len(self.imgs) - 1):
            match = Flann_Match(self.descriptors[i], self.descriptors[i + 1])
            self.matchs.append(match)
            kpsA=self.kpts[i]
            kpsB=self.kpts[i+1]
            if len(match) > 4:
                # 获取匹配对的点坐标
                ptsA = np.float32([kpsA[i] for (_, i) in match])
                ptsB = np.float32([kpsB[i] for (i, _) in match])
                # 计算视角变换矩阵
                H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
                imag=warp(self.imgs[i],self.imgs[i+1],H)
                plt.imshow(imag)
                plt.show()
            # print(H)
            # print(status)

    def match(self):
        pass


if __name__ == '__main__':
    ima = cv2.imread("data/3.jpg")
    imb = cv2.imread("data/4.jpg")
    imgs = []
    imgs.append(ima)
    imgs.append(imb)
    pano=PanoramaStitcher(imgs)
    pano.run()