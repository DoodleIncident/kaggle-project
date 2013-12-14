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

def compute_error():
