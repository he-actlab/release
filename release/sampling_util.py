"""
sampling_util.py: function to help reconstruct samples
"""

__author__     = "Byung Hoon Ahn"
__email__      = "bhahn@eng.ucsd.edu"

import numpy as np
import random

def get_samples(points, dims, effective_dims, centroids, cluster):
    samples = []
    for c in range(0, len(centroids)):
        # find members
        members = []
        for p in range(0, len(points)):
            if cluster[p] == c:
                members.append(points[p])

        # calculate other coordinates
        sample_dims = {}
        if len(members) == 0:
            for d in range(0, len(points[0])):
                sample_dims[d] = range(0, dims[d])
        else:    
            for d in range(0, len(points[0])):
                if d in effective_dims:
                    continue
                
                if d not in sample_dims:
                    sample_dims[d] = []

                for m in members:
                    sample_dims[d].append(m[d])

        # construct sample
        sample = []
        for d in range(0, len(points[0])):
            if d in effective_dims:
                sample.append(centroids[c][list(effective_dims).index(d)])
                continue
            
            # deterministic
            #sample.append(max(set(sample_dims[d]), key=sample_dims[d].count))

            # sampling
            sample.append(random.sample(sample_dims[d], 1)[0])

        done = False
        while done is False:
            if sample not in samples:
                samples.append(sample)
                done = True
            else:
                sample = random.sample(points, 1)[0]

    return samples
