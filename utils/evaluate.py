import numpy as np

from scipy.spatial import distance


# the original evaluation function
def fx_calc_map_label(image, text, label, k=0, dist_method='COS', sample_num=None):
    '''
    Add sample method to speed up the evaluation processing.
    '''
    if sample_num and len(image) > sample_num:
        random_sample = np.random.randint(0, len(image), sample_num)
        image = image[random_sample]
        text = text[random_sample]
        label = label[random_sample]

    if dist_method == 'L2':
        dist = distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = distance.cdist(image, text, 'cosine')

    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if check_label(label[i], label[order[j]]):
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)


# check label similarity
def check_label(x, y):
    '''
    Used for solve multi-label problem also support one-hot label
    '''
    # flag = bool(np.sum(np.logical_and(x, y)))

    # a speed up version
    flag = np.dot(x, y) > 0
    return flag


# the original evaluation function
def fx_calc_map_label_with_diff_label(image, text, label_q, label_r, k=0, dist_method='COS'):
    '''
    Add sample method to speed up the evaluation processing.
    '''

    if dist_method == 'L2':
        dist = distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = distance.cdist(image, text, 'cosine')
    elif dist_method == 'Ham':
        dist = calc_hammingDist(image, text)

    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if check_label(label_q[i], label_r[order[j]]):
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH
