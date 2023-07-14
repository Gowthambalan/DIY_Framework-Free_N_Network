import math
from copy import deepcopy


def mat_mul(a, b):
    i, j = len(a), len(a[0])
    n, k = len(b), len(b[0])

    assert n == j
    temp = [[None for _ in range(k)] for _ in range(i)]
    for w in range(i):
        for h in range(k):
            temp[w][h] = 0
            for r in range(j):
                temp[w][h] += a[w][r] * b[r][h]
    return temp


def mat_add(a, b):
    i, j = len(a), len(a[0])
    n, k = len(b), len(b[0])

    assert i == n and j == k
    temp = [[None for _ in range(j)] for _ in range(i)]
    for w in range(i):
        for h in range(j):
            temp[w][h] = a[w][h] + b[w][h]
    return temp


def add_scalar(a, scalar):
    i, j = len(a), len(a[0])
    temp = [[None for _ in range(j)] for _ in range(i)]
    for w in range(i):
        for h in range(j):
            temp[w][h] = a[w][h] + scalar
    return temp


def element_wise_mul(a, b):
    i, j = len(a), len(a[0])
    n, k = len(b), len(b[0])

    assert i == n and j == k
    temp = [[None for _ in range(j)] for _ in range(i)]
    for w in range(i):
        for h in range(j):
            temp[w][h] = a[w][h] * b[w][h]
    return temp


def element_wise_rev(a):
    i, j = len(a), len(a[0])
    temp = [[None for _ in range(j)] for _ in range(i)]
    for w in range(i):
        for h in range(j):
            temp[w][h] = 1 / a[w][h]
    return temp


def mat_sqrt(a):
    i, j = len(a), len(a[0])
    temp = [[None for _ in range(j)] for _ in range(i)]
    for w in range(i):
        for h in range(j):
            temp[w][h] = math.sqrt(a[w][h])
    return temp


def transpose(a):
    i, j = len(a), len(a[0])
    temp = [[None for _ in range(i)] for _ in range(j)]
    for w in range(j):
        for h in range(i):
            temp[w][h] = a[h][w]
    return temp


def rescale(a, scale):
    i, j = len(a), len(a[0])
    temp = [[None for _ in range(j)] for _ in range(i)]
    for w in range(i):
        for h in range(j):
            temp[w][h] = a[w][h] * scale
    return temp


def batch_sum(a):
    i, j = len(a), len(a[0])
    temp = [[0 for _ in range(j)]]
    for w in range(i):
        for h in range(j):
            temp[0][h] += a[w][h]
    return temp


def batch_mean(a):
    i, j = len(a), len(a[0])
    temp = [[0 for _ in range(j)]]
    for w in range(i):
        for h in range(j):
            temp[0][h] += (a[w][h] / i)
    return temp


def batch_var(a, mu):
    i, j = len(a), len(a[0])
    temp = [[0 for _ in range(j)]]
    for w in range(i):
        for h in range(j):
            temp[0][h] += ((a[w][h] - mu[0][h]) ** 2) / i
    return temp


def equal_batch_size(a, b):
    w0, w1 = len(a), len(b)
    if w0 < w1:
        w = w1
        temp = deepcopy(a)
        x = a
    else:
        w = w0
        temp = deepcopy(b)
        x = b
    while len(temp) < w:
        temp.append(x[0])
    if w0 < w1:
        return temp, b
    else:
        return a, temp


if __name__ == "__main__":
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    print(mat_mul(a, b))
    print(mat_add(a, b))
    print(element_wise_mul(a, b))
    print(transpose(a))
    print(rescale(a, 1 / 5))
