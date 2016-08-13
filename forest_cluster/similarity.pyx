#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.int_t DTYPE_t
ctypedef np.intp_t ITYPE_t


cdef inline double dist(DTYPE_t* x1, DTYPE_t* x2,
                         ITYPE_t size) nogil except -1:
    cdef int n_eq = 0
    cdef ITYPE_t j
    for j in range(size):
        n_eq += (x1[j] == x2[j])
    return n_eq * 1. / size


def matching_dist(np.ndarray[DTYPE_t, ndim=1] X1, np.ndarray[DTYPE_t, ndim=1] X2):
    cdef ITYPE_t size = X1.shape[0]

    return dist(<DTYPE_t*> X1.data, <DTYPE_t*> X2.data, size)
