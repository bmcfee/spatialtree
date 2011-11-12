#!/usr/bin/env python
'''
CREATED:2011-11-12 09:39:09 by Brian McFee <bmcfee@cs.ucsd.edu>
 
Spatial tree demo for dictionary-type data

Any object X which implements a mapping to vector data (X[key] -> numpy array) 
may be used here.

'''

import numpy
from spatialtree import spatialtree

N = 5000
D = 20

# A random projection matrix, for funsies
P = numpy.random.randn(D, D)

# For convenience, let's define a random point generator
def newpoint():
    return numpy.dot(numpy.random.randn(D), P)

print 'Generating data...' 
# Now, let's populate the dictionary with N random points
X = dict()

for i in xrange(N):
    # Let's use string-valued keys
    X['%04x' %i] = newpoint()
    pass

# Let's make a few distinguished points
X['Alice']  = newpoint()
X['Bob']    = newpoint()
X['Carol']  = newpoint()

print 'done.'

# Construct a tree.  Let's use a 2-means tree with spill percentage 0.3
print 'Building tree...'
T = spatialtree(X, rule='2-means', spill=0.3)
print 'done.'


# Show some stats
print '# items in tree    : ', len(T)
print 'Dimensionality     : ', T.getDimension()
print 'Height of tree     : ', T.getHeight()
print 'Spill percentage   : ', T.getSpill()
print 'Split rule         : ', T.getRule()

# Let's find the nearest neighbors of bob:
knn_bob = T.k_nearest(X, k=10, index='Bob')
print 'KNN(Bob)           : ', knn_bob

# Or of a random vector:
knn_random = T.k_nearest(X, k=10, vector=newpoint())
print 'KNN(random)        : ', knn_random


# With dictionary-type data, we add to the tree as well
X['Dave'] = newpoint()
T.update({'Dave': X['Dave']})

# For retrieval purposes, the new point will have to live in X from then onward
knn_dave = T.k_nearest(X, k=10, index='Dave')
print 'KNN(Dave)          : ', knn_dave
