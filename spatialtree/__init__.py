#!/usr/bin/env python
'''
CREATED:2011-11-11 13:53:46 by Brian McFee <bmcfee@cs.ucsd.edu>

Implementation of spatial trees:
    * Max-variance KD
    * PCA tree
    * 2-means tree
    * RP tree

Also with spill support
'''

import numpy
import scipy.stats
import random

class spatialtree(object):

    def __init__(self, data, **kwargs):
        '''
        T = spatialtree(    data, 
                            rule='pca', 
                            height=H, 
                            spill=0.0, 
                            indices=(index1, index2,...), 
                            steps_2means=1000,
                            samples_rp=10)
                            

        Required arguments:
            data:           n-by-d data matrix (numpy.ndarray), one point per row
                            alternatively, may be a dict of vectors

        Optional arguments:
            rule:           must be one of 'kd', 'pca', '2-means', 'rp'

            height>0:       maximum-height to build the tree
                            default: ceiling(log2(n) - 7)

            spill:          how much overlap to allow between children at split
                            must lie in range [0,1)

            indices:        list of keys/indices to store in this (sub)tree
                            default: 0:n-1, or data.keys()

        Split-specific:
            steps_2means:   minimum number of steps for building the 2-means tree

            samples_rp:     number of directions to consider for each RP split
        '''

        # Default values
        if 'indices' not in kwargs:
            if isinstance(data, dict):
                kwargs['indices']   = data.keys()
            else:
                kwargs['indices']   = range(len(data))
            pass

        n = len(kwargs['indices'])

        if 'rule' not in kwargs:
            kwargs['rule']          = 'pca'
            pass

        kwargs['rule'] = kwargs['rule'].lower()

        if 'height' not in kwargs:
            kwargs['height']        = max(0, numpy.ceil(numpy.log2(n) - 7))
            pass

        if 'spill' not in kwargs:
            kwargs['spill']         = 0.0
            pass

        if spill < 0.0 or spill >= 1.0:
            raise ValueError('spill=%.2e, must lie in range [0,1)' % spill)

        if kwargs['rule'] == '2-means' and 'steps_2means' not in kwargs:
            kwargs['steps_2means']  = 1000
            pass

        if kwargs['rule'] == 'rp' and 'samples_rp' not in kwargs:
            kwargs['samples_rp']    = 10

        # All information is now contained in kwargs, we may proceed
        

        # Store bookkeeping information
        self.__indices      = set(kwargs['indices'])
        self.__splitRule    = kwargs['rule']
        self.__spill        = kwargs['spill']
        self.__children     = (None, None)
        self.__w            = None
        self.__thresholds   = (None, None)
        self.__n            = len(self.__indices)
        for x in self.__indices:
            self.__d = len(data[x])
            break
        del x

        # Split the node
        self.split(data, **kwargs)

        pass

    def split(self, data, **kwargs):

        # First, find the split rule
        if kwargs['rule'] == 'pca':
            splitF  =   self.__PCA
        elif kwargs['rule'] == 'kd':
            splitF  =   self.__KD
        elif kwargs['rule'] == '2-means':
            splitF  =   self.__2means
        elif kwargs['rule'] == 'rp':
            splitF  =   self.__RP
        else:
            raise ValueError('Unsupported split rule: %s' % kwargs['rule'])

        # If the height is 0, we don't need to split
        if kwargs['height'] == 0:
            return

        if kwargs['height'] < 1:
            raise ValueError('spatialtree.split() called with height<0')

        # Compute the split direction 
        self.__w = splitF(data, **kwargs)

        # Project onto direction
        wx = {}
        for i in self.__indices:
            wx[i] = numpy.dot(self.__w, X[i])
            pass

        # Compute the bias points
        self.__thresholds = scipy.stats.mstats.mquantiles(wx.values, [0.5 - self.__spill/2, 0.5 + self.__spill/2])

        # Partition the data
        left_set    = set()
        right_set   = set()

        for (i, val) in wx.iteritems():
            if val >= self.__thresholds[0]:
                right_set.add(i)
            if val < self.__thresholds[-1]:
                left_set.add(i)
            pass
        del wx

        # Construct the children
        kwargs['height']    -= 1

        kwargs['indices']   = left_set
        self.__children[0]  = spatialtree(data, **kwargs)
        del left_set

        kwargs['indices']   = right_set
        self.__children[1]  = spatialtree(data, **kwargs)
        del right_set

        # Done
        pass

    # SPLITTING RULES

    def __PCA(self, data, **kwargs):
        # first moment
        moment_1 = numpy.zeros(self.__d)

        # second moment
        moment_2 = numpy.zeros((self.__d, self.__d))

        # Compute covariance matrix
        for i in self.__indices:
            moment_1 += data[i]
            moment_2 += numpy.outer(data[i], data[i])
            pass

        # the mean
        moment_1    /= self.__n   

        # the covariance
        sigma       = (moment_2 - (self.__n * numpy.outer(moment_1, moment_1))) / (self.__n - 1.0)

        # eigendecomposition
        (l, v)      = numpy.linalg.eigh(sigma)
        
        # top eigenvector
        w           = v[:, numpy.argmax(l)]
        return w

    def __KD(self, data, **kwargs):
        moment_1 = numpy.zeros(self.__d)
        moment_2 = numpy.zeros(self.__d)

        for i in self.__indices:
            moment_1 += data[i]
            moment_2 += data[i] ** 2
            pass

        # mean
        moment_1    /= self.__n

        # variance
        sigma       = (moment_2 - (self.__n * moment_1**2)) / (self.__n - 1.0)

        # the coordinate of maximum variance
        w           = numpy.zeros(self.__d)
        w[numpy.argmax(sigma)] = 1
        return w

    def __2means(self, data, **kwargs):
        def D(u,v):
            return numpy.sum( (u-v)**2 )

        centers     = numpy.zeros( (2, self.__d) )
        counters    = [0] * 2

        index       = list(self.__indices)
        count       = 0
        num_steps   = max(self.__n, kwargs['steps_2means'])

        while True:
            # Randomly permute the index
            random.shuffle(index)
            
            for i in index:
                # Find the closest centroid
                j_min = numpy.argmin([D(data[i], mu) * c / (1.0 + c) for (mu, c) in zip(centers, counters)])

                centers[j_min,:] = (centers[j_min,:] * counters[j_min] + data[i]) / (counters[j_min]+1)
                counters[j_min] += 1

                count += 1
                if count > num_steps:
                    break
                pass
            if count > num_steps:
                break

        w = centers[0,:] - centers[1,:]

        w /= numpy.sqrt(numpy.sum(w**2))
        return w


    def __RP(self, data, **kwargs):
        k = kwargs['samples_rp']
        # sample directions from the unit sphere
        W   = numpy.random.randn( (k, self.__d]) )

        for i in xrange(k):
            W[i,:] /= numpy.sqrt(numpy.sum(W[i,:]**2))

        # Find the direction that maximally spreads the data:

        min_val = numpy.inf * numpy.ones(k)
        max_val = -numpy.inf * numpy.ones(k)

        for i in self.__indices:
            Wx      = numpy.dot(W, data[i])
            min_val = numpy.minimum(min_val, Wx)
            max_val = numpy.maximum(max_val, Wx)
            pass

        return W[numpy.argmax(max_val - min_val),:]
