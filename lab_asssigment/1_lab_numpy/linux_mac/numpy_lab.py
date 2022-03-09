import numpy as np


def n_size_ndarray_creation(n, dtype=int):
    np_array = np.arange(n ** 2).reshape(n, n) ** 2
    print(np_array)
    print(np_array.size)
    print(np_array.shape)
    print(np_array.ndim)
    return np_array


def zero_or_one_or_empty_ndarray(shape, type=0, dtype=int):
    np_array = None
    if type == 0:  # zero
        np_array = np.zeros(shape=shape, dtype=dtype)
    elif type == 1:  # ones
        np_array = np.ones(shape=shape, dtype=dtype)
    elif type == 99:  # empty
        np_array = np.empty(shape=shape, dtype=dtype)
    else:
        print(type)
        pass
    return np_array


def change_shape_of_ndarray(X, n_row):
    if n_row == 1:
        np_array = X.flatten()
    else:
        np_array = X.reshape(n_row, -1)
    return np_array


def concat_ndarray(X_1, X_2, axis):
    # np_array = np.concatenate((X_1, X_2), axis=axis)
    # get max ndim between X_1 and X_2.

    np_array = None

    # Reshape to matrix if each is vector.
    if X_1.ndim == 1:
        X_1 = X_1.reshape(1, -1)
        # print(X_1)
        # print(X_1.shape)
        # print(X_1.shape[0])
        # print(X_1.shape[1])
    if X_2.ndim == 1:
        X_2 = X_2.reshape(1, -1)
        # print(X_2)

    if axis == 0:  # the number of column should be the same
        if X_1.shape[1] != X_2.shape[1]:
            np_array = False
        else:
            np_array = np.vstack((X_1, X_2))  # Up Down
    elif axis == 1:  # the number of row should be the same
        if X_1.shape[0] != X_2.shape[0]:
            np_array = False
        else:
            np_array = np.hstack((X_1, X_2))  # Left Right
    else:
        pass

    return np_array


def normalize_ndarray(X, axis=99, dtype=np.float32):
    Z = None
    if axis == 0:  # rox, axis = 0
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        print(X)
        print(x_mean)
        print(x_std)
        Z = (X - x_mean) / x_std
    elif axis == 1:  # column, axis = 1
        x_mean = np.mean(X, axis=1).reshape(X.shape[0], -1)
        x_std = np.std(X, axis=1).reshape(X.shape[0], -1)
        # print(X)
        # print(x_mean)
        # print(x_std)
        Z = (X - x_mean) / x_std
    elif axis == 99:
        x_mean = np.mean(X)
        x_std = np.std(X)
        # print(x_mean)
        # print(x_std)
        Z = (X - x_mean) / x_std
    else:
        pass
    return Z


def save_ndarray(X, filename="test.npy"):
    # print(X)
    np.save(filename, X)

    # load_array = np.load(filename)
    # print(load_array)


def boolean_index(X, condition):
    # return np.where(X[eval(str("X") + condition)])
    return np.where(eval(str("X") + condition))


def find_nearest_value(X, target_value):
    index = np.argmin(abs(X - target_value))
    return X[index]


def get_n_largest_values(X, n):
    X.sort()
    X = X[::-1]
    return X[0:n]


# def main():
# n_size_ndarray_creation(10)
# ret = zero_or_one_or_empty_ndarray((4,4), 99, float)
# print(ret)
# ret = zero_or_one_or_empty_ndarray((3,3,3), 99, int)
# print(ret)
# ret = zero_or_one_or_empty_ndarray((10), 0)
# print(ret)
# ret = zero_or_one_or_empty_ndarray((10), 1)
# print(ret)

# test_matrix = [1,2,3,4,5,6,7,8,9,10]
# ret = change_shape_of_ndarray(np.array(test_matrix), 1)
# print(ret)

# a = np.array([[1,2,3],[7,8,9]])
# b = np.array([[4,5,6],[10,11,12]])

#    a = np.array([[1, 2], [3, 4]])
#    b = np.array([[5, 6]])

#    ret = concat_ndarray(a, b, 0)
#    print(ret)

#    ret = concat_ndarray(a, b, 1)
#    print(ret)

#    a = np.array([1, 2])
#    b = np.array([4, 5, 6])

#    ret = concat_ndarray(a, b, 1)
#    print(ret)

#    ret = concat_ndarray(a, b, 0)
#    print(ret)

#    a = np.array([[1],[2],[3]])
#    b = np.array([[4],[5],[6]])

#    ret = concat_ndarray(a, b, 1)
#    print(ret)

# X = np.arange(12, dtype=np.float32).reshape(6, 2)

# ret = normalize_ndarray(X)
# print(ret)

# ret = normalize_ndarray(X, 0)
# print(ret)

# ret = normalize_ndarray(X, 1)
# print(ret)

# X = np.arange(32, dtype=np.float32).reshape(4, -1)
# filename = "test.npy"
# save_ndarray(X, filename)

# X = np.arange(32, dtype=np.float32).reshape(4, -1)
# print(X)
# ret = boolean_index(X, "== 3")
# print(ret)

# X = np.arange(32, dtype=np.float32)
# print(X)
# ret = boolean_index(X, "> 6")
# print(ret)

# X = np.random.uniform(0, 1, 100)
# target_value = 0.3
# X = np.arange(32, dtype=int)
# target_value = 10
# ret = find_nearest_value(X, target_value)
# print(ret)

# X = np.random.uniform(0, 1, 100)
# ret = get_n_largest_values(X, 3)
# print(ret)

# ret = get_n_largest_values(X, 5)
# print(ret)


# if __name__ == "__main__":
# main()

