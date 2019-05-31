import argparse
import sys
import os
import glob
import numpy as np
from PIL import Image
from itertools import combinations


def get_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='folder with images')
    return parser.parse_args().path


def hist(im, bin_numb=50):
    return np.histogram(im, bins=bin_numb)[0]


def find_similar(path, threshold=0.81):
    '''
    Find duplicate, similar and modified images
    by computing histogram overlay of two images
    for every channel.

    input  - path to directory containing images
    output - list of similar images
    '''
    files = glob.glob(path + '/*')
    images = []
    similar = []

    for i in files[:]:
        try:
            images.append(np.array(Image.open(i).resize((48, 48))))
        except:
            files.remove(i)

    file_hists = {}
    # dictionary file name - image
    for key, val in enumerate(images):
        file_hists[files[key]] = np.array([hist(val[:, :, 0], bin_numb=60),
                                           hist(val[:, :, 1], bin_numb=60), hist(val[:, :, 2], bin_numb=60)])

    for i, j in combinations(range(len(files)), 2):
        chanels_overlay = np.zeros((3))

        for k in range(3):
            first = np.minimum(file_hists[files[i]][k], file_hists[files[j]][k])
            chanels_overlay[k] = np.true_divide(np.sum(first) + 1e-10, np.sum(file_hists[files[j]][k]) + 1e-10)
        # avoid division 0 / 0 if duplicates
        mean_overlay = np.mean(chanels_overlay)
        # take mean overlay over all three channels
        if mean_overlay > threshold:
            similar.append((files[i], files[j]))
    return similar


def run():
    path = get_path()

    if not os.path.exists(path):
        print(path, " does not exist")

    else:
        similar = find_similar(path)
        if len(similar) == 0:
            print('Nothing found.')
        for i, j in similar:
            print(i, j)


if __name__ == '__main__':
    run()
