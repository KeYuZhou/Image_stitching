import numpy as np
import cv2 as cv



def Flann_Match(des1,des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good=[]
    for m in matches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            # 存储两个点在featuresA, featuresB中的索引值
            good.append((m[0].trainIdx, m[0].queryIdx))
    return good
    # Need to draw only good matches, so create a mask
    # matchesMask = [[0, 0] for i in range(len(matches))]
    # # ratio test as per Lowe's paper
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.7 * n.distance:
    #         matchesMask[i] = [1, 0]
    # return matchesMask