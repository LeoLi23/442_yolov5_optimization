import cv2
import numpy as np
import math


def dark_channel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def get_atm_light(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def get_transmission(im, A, sz, omega=0.95):
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * dark_channel(im3, sz)
    return transmission


def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b
    return q


def get_refined_transmission(I, p, r=60, eps=0.0001):
    I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I_gray = I_gray.astype(np.float64) / 255
    p = p.astype(np.float64) / 255
    q = guided_filter(I_gray, p, r, eps)

    return (q * 255).astype(np.uint8)


def remove_haze(img, sz=15, omega=0.95, r=60, eps=0.0001):
    I = img.astype('float64') / 255
    dark = dark_channel(I, sz)
    A = get_atm_light(I, dark)
    raw_transmission = get_transmission(I, A, sz, omega)
    refined_transmission = get_refined_transmission(img, raw_transmission, r, eps)
    t = refined_transmission.astype(np.float64) / 255

    J = np.empty(img.shape, img.dtype)
    for ind in range(0, 3):
        J[:, :, ind] = (I[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return (J * 255).astype(np.uint8)
