import copy
import os
import math
import cv2
import sys
import numpy as np
from random import randint

def parse_args():
    directory = sys.argv[1] + '//'
    img_paths = []
    for file_name in os.listdir(directory):
        if '.jpg' in file_name:
            img_paths.append(directory + file_name)
    img_paths.sort()
    return directory, img_paths


def getSSD(descriptor1,descriptor2):
    ssd = np.sum(np.square(np.subtract(descriptor1,descriptor2)))
    return ssd


def findMatches(descriptor1,descriptor2):
    matches = []
    for i in range(len(descriptor1)):
        curr_matches = []
        for j in range(len(descriptor2)):
            curr_ssd = getSSD(descriptor1[i],descriptor2[j])
            match = cv2.DMatch(j,i,curr_ssd)
            #match = cv2.DMatch(trainIdx = i,queryIdx = j,distance = curr_ssd)
            curr_matches.append(match)
        curr_matches = sorted(curr_matches, key = lambda x:x.distance)
        if curr_matches[0].distance < 0.07*curr_matches[1].distance:
            matches.append(curr_matches[0])
    return matches


def getHomographyMatrix(matches,kps):
    A = []
    for match in matches:
        x, y = kps[0][match.trainIdx].pt
        u, v = kps[1][match.queryIdx].pt
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    H = np.ndarray.reshape(Vh[8], (3, 3))
    H = (1/H.item(8)) * H
    return H


def getDistance(x1, y1, x2, y2, homography):
    point1 = np.transpose(np.matrix([x1, y1, 1]))
    point2 = np.transpose(np.matrix([x2, y2, 1]))
    transformedPoint = np.dot(homography, point1)
    if (transformedPoint.item(2) and transformedPoint.item(2) != 0):
        transformedPoint = (1 / transformedPoint.item(2)) * transformedPoint

    distance = transformedPoint - point2
    return np.linalg.norm(distance)


def getRandomFourMatches(matches):
    randomMatches = []
    while len(randomMatches) != 4:
        randomMatch = matches[randint(0, len(matches)-1)]
        if randomMatch not in randomMatches:
            randomMatches.append(randomMatch)
    return randomMatches


def implementRansac(matches, kps):
    max_inliers = 0
    final_H = []
    for i in range(len(matches)*100):
        randomMatches = getRandomFourMatches(matches)
        curr_H = getHomographyMatrix(randomMatches,kps)
        inliers = 0
        for match in matches:
            x1,y1 = kps[0][match.trainIdx].pt
            x2,y2 = kps[1][match.queryIdx].pt
            distance = getDistance(x1,y1,x2,y2,curr_H)
            if distance < 1:
                inliers += 1
        if inliers > len(matches)*0.90:
            #print("best H found")
            final_H = curr_H
            break
        if inliers > max_inliers:
            #print("adjusted H found")
            max_inliers = inliers
            final_H = curr_H
    #print(max_inliers)
    return final_H


def warpPerspective(img1_, img2_, final_H):
    img1_height = img2_.shape[1]
    img1_width = img2_.shape[0]
    img2_height = img1_.shape[1]
    img2_width = img1_.shape[0]

    image_frame1 = np.float32([[0, 0], [0, img1_width], [img1_height, img1_width], [img1_height, 0]]).reshape(-1, 1, 2)
    image_frame2 = np.float32([[0, 0], [0, img2_width], [img2_height, img2_width], [img2_height, 0]]).reshape(-1, 1, 2)
    image_frame2_transformed = cv2.perspectiveTransform(image_frame2, final_H)
    final_image_frame = np.vstack((image_frame1, image_frame2_transformed))
    [minx, miny] = np.int32(final_image_frame.min(axis=0).flatten())
    [maxx, maxy] = np.int32(final_image_frame.max(axis=0).flatten())
    translation_dist = [-minx, -miny]
    h_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    anchor_img = cv2.warpPerspective(img1_, h_translation.dot(final_H), (maxx - minx, maxy - miny))
    anchor_img[translation_dist[1]:img1_width + translation_dist[1], translation_dist[0]:img1_height + translation_dist[0]] = img2_

    return anchor_img


def stitch(img1_, img2_path):
    img1 = cv2.cvtColor(img1_, cv2.IMREAD_GRAYSCALE)

    img2_ = cv2.imread(img2_path)
    img2 = cv2.cvtColor(img2_, cv2.IMREAD_GRAYSCALE)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    print(f"kp1:{len(kp1)},kp2:{len(kp2)}")
    
    kps = [kp1, kp2]
    matches = findMatches(des1, des2)
    print(len(matches))
    if len(matches) > 4:
        final_homography = implementRansac(matches, kps)
        anchor_img = warpPerspective(img1_, img2_, final_homography)
        return anchor_img, True
    else:
        return img2_path, False


def get_panorama(img_paths):
    anchorImg_path = img_paths.pop()
    anchor_img = cv2.imread(anchorImg_path)
    while len(img_paths) > 0:
        img, flag = stitch(anchor_img, img_paths.pop())
        if flag:
            anchor_img = img
        #else:
            #img_paths.append(img)
    return anchor_img


directory, img_paths = parse_args()

panorama = get_panorama(img_paths)

cv2.imwrite(directory+"panorama.jpg", panorama)
