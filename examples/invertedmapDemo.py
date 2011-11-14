#!/usr/bin/env python
'''
CREATED:2011-11-13 16:00:16 by Brian McFee <bmcfee@cs.ucsd.edu>

Demo of inverted maps for lightweight static indexing
'''

import numpy
from spatialtree import spatialtree
from spatialtree import invertedmap


# First, create a random data matrix
N = 5000
D = 20

X = numpy.random.randn(N,D)


# Apply a random projection so the data's not totally boring
P = numpy.random.randn(D, D)

X = numpy.dot(X, P)

# Construct a tree.  This time, we'll use a random projection tree of height 10

print 'Building tree...'
T = spatialtree(X, rule='rp', height=10)
print 'done.'

# Show some useful information about the tree
print '# items in tree    : ', len(T)
print 'Dimensionality     : ', T.getDimension()
print 'Height of tree     : ', T.getHeight()
print 'Spill percentage   : ', T.getSpill()
print 'Split rule         : ', T.getRule()

#
# By default, spatialtree retains index information at every level 
# throughout the tree.  This facilitates pruning and other dynamic 
# modifications to the tree.  However, if your data set and tree
# are static (after construction of the tree), then we can make a more
# space-efficient data structure by using an inverted map.
#
# The invertedmap structure only stores the relationship between items
# and the leaves of the tree from which it is constructed.

print 'Converting tree T to inverted map I...'
I = invertedmap(T)
print 'done.'

# Some stats:
print '# items in the map : ', len(I)
print '# leaf sets        : ', I.numSets()

# It can answer k_nearest queries just like a spatialtree
knn_t = T.k_nearest(X, k=10, index=499)
print 'KNN approx (tree)  : ', knn_t

# The imap discards all vector data from the structure,
# so it can only answer queries by index.
knn_i = I.k_nearest(X, k=10, index=499)
print 'KNN approx (imap)  : ', knn_i

# You can also delete items from the inverted map, but
# new nodes cannot be added.

I.remove(499)
