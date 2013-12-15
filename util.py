import numpy
import scipy.sparse as sp

def save_sparse_matrix(filename,x):
    x_coo=x.tocoo()
    row=x_coo.row
    col=x_coo.col
    data=x_coo.data
    shape=x_coo.shape
    numpy.savez(filename,row=row,col=col,data=data,shape=shape)

def load_sparse_matrix(filename):
    y=numpy.load(filename)
    z=sp.coo_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
    return z

def sparser(flurp):
    herp = []
    for i in flurp:
        derp = []
        for idx,val in enumerate(i):
            if val > 0:
                derp.append(idx)
        herp.append(derp)
    return herp

def denser(flurp,a):
    herp = []
    for i in flurp:
        derp = []
        for j in range(0,a):
            if j in i:
                derp.append(1)
            else:
                derp.append(0)
        herp.append(derp)
    return herp

def list_of_tuples_to_tuple_of_lists(flurp):
    herp = ()
    for i in flurp:
        derp = []
        for j in i:
            derp.append(j)
        herp += (derp,)
    return herp

def list_of_tuples_to_2d_list(flurp):
    herp = []
    for i in flurp:
        derp = []
        for j in i:
            derp.append(j)
        herp.append(derp)
    return herp

def compute_error(a,b,c):
    acc = 0
    for i,j in zip(a,b):
        for k in range(0,c):
            if k in i and k in j:
                acc += 1
            if k not in i and k not in j:
                acc += 1
    return acc
