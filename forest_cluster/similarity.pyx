import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.int_t DTYPE_t
ctypedef np.intp_t ITYPE_t


cdef inline DTYPE_t dist(int* x1, int* x2,
                         ITYPE_t size) nogil except -1:
    cdef int n_eq = 0
    cdef ITYPE_t j
    for j in range(size):
        n_eq += (x1[j] == x2[j])
    return n_eq * 1. / size


def matching_dist(np.ndarray[int, ndim=1] X1, np.ndarray[int, ndim=1] X2):
    cdef ITYPE_t size = X1.shape[0]

    return dist(<int*> X1.data, <int*> X2.data, size)
