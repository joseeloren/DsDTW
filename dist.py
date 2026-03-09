#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy, pdb
from fastdtw import fastdtw as dtw #https://github.com/slaypni/fastdtw/issues
from matplotlib import pyplot as plt
from soft_dtw_cuda import SoftDTW
import torch
from joblib import Parallel, delayed
import multiprocessing

def process_user(idx, feat_seq, ng, nf, num_g, num_f):
    feat_a = feat_seq[0:ng]
    feat_p = feat_seq[(ng+nf):(num_g+nf)]
    feat_n = feat_seq[(num_g+nf):]
        
    dist_p = numpy.zeros((num_g-ng, ng))
    dist_n = numpy.zeros((num_f, ng))
    dist_temp = numpy.zeros((ng, ng))
    
    for i in range(ng):
        fp = feat_a[i]
        fps = numpy.sum(fp, axis=1)
        fp = numpy.delete(fp, numpy.where(fps == 0)[0], axis=0) 
        for j in range(i+1, ng): 
            fa = feat_a[j]
            fas = numpy.sum(fa, axis=1)
            fa = numpy.delete(fa, numpy.where(fas == 0)[0], axis=0) 
            dist, path = dtw(fp, fa, radius=2, dist=1)
            dist = dist / (fp.shape[0] + fa.shape[0])
            dist_temp[i, j] = dist
    for i in range(num_g-ng):
        fp = feat_p[i]
        fps = numpy.sum(fp, axis=1)
        fp = numpy.delete(fp, numpy.where(fps == 0)[0], axis=0) 
        for j in range(ng):
            fa = feat_a[j]
            fas = numpy.sum(fa, axis=1)
            fa = numpy.delete(fa, numpy.where(fas == 0)[0], axis=0) 
            dist, path = dtw(fp, fa, radius=2, dist=1)
            dist = dist / (fp.shape[0] + fa.shape[0])
            dist_p[i, j] = dist
    for i in range(num_f):
        fn = feat_n[i]
        fns = numpy.sum(fn, axis=1)
        fn = numpy.delete(fn, numpy.where(fns == 0)[0], axis=0) 
        for j in range(ng):
            fa = feat_a[j]
            fas = numpy.sum(fa, axis=1)
            fa = numpy.delete(fa, numpy.where(fas == 0)[0], axis=0) 
            dist, path = dtw(fn, fa, radius=2, dist=1)
            dist = dist / (fn.shape[0] + fa.shape[0])
            dist_n[i, j] = dist
            
    return dist_p, dist_n, dist_temp

def dist_seq(FEAT_SEQ, ng, nf, num_g, num_f):
    print("Calculating DTW distance... (Parallel CPU)")
    num_cores = max(1, multiprocessing.cpu_count() // 6) # Divide by 5-6 concurrent seeds to avoid RAM OOM
    results = Parallel(n_jobs=num_cores)(delayed(process_user)(idx, feat_seq, ng, nf, num_g, num_f) for idx, feat_seq in enumerate(FEAT_SEQ))
    
    DIST_P = [r[0] for r in results]
    DIST_N = [r[1] for r in results]
    DIST_TEMP = [r[2] for r in results]
    
    DIST_P = numpy.concatenate(DIST_P, axis=0)
    DIST_N = numpy.concatenate(DIST_N, axis=0)
    DIST_TEMP = numpy.concatenate(DIST_TEMP, axis=0)

    return DIST_P, DIST_N, DIST_TEMP

def process_rf_user(idx, feat_a, feat_p, ng, FEAT_A, FEAT_P):
    feat_n = []
    for i in range(len(FEAT_A)):
        if i!=idx:
            feat_n.append(FEAT_P[i][2])
            
    dist_p = numpy.zeros((feat_p.shape[0], ng))
    dist_n = numpy.zeros((len(feat_n), ng))
    dist_temp = numpy.zeros((ng, ng))
    
    for i in range(ng):
        fp = feat_a[i]
        fps = numpy.sum(fp, axis=1)
        fp = numpy.delete(fp, numpy.where(fps == 0)[0], axis=0) 
        for j in range(i+1, ng): 
            fa = feat_a[j]
            fas = numpy.sum(fa, axis=1)
            fa = numpy.delete(fa, numpy.where(fas == 0)[0], axis=0) 
            dist, path = dtw(fp, fa, radius=2, dist=1) 
            dist = dist / (fp.shape[0] + fa.shape[0])
            dist_temp[i, j] = dist
            
    for i in range(feat_p.shape[0]):
        fp = feat_p[i]
        fps = numpy.sum(fp, axis=1)
        fp = numpy.delete(fp, numpy.where(fps == 0)[0], axis=0) 
        for j in range(ng):
            fa = feat_a[j]
            fas = numpy.sum(fa, axis=1)
            fa = numpy.delete(fa, numpy.where(fas == 0)[0], axis=0) 
            dist, path = dtw(fp, fa, radius=2, dist=1) 
            dist = dist / (fp.shape[0] + fa.shape[0])
            dist_p[i, j] = dist
            
    for i in range(len(feat_n)):
        fn = feat_n[i]
        fns = numpy.sum(fn, axis=1)
        fn = numpy.delete(fn, numpy.where(fns == 0)[0], axis=0) 
        for j in range(ng):
            fa = feat_a[j]
            fas = numpy.sum(fa, axis=1)
            fa = numpy.delete(fa, numpy.where(fas == 0)[0], axis=0) 
            dist, path = dtw(fn, fa, radius=2, dist=1) 
            dist = dist / (fn.shape[0] + fa.shape[0])
            dist_n[i, j] = dist        
            
    return dist_p, dist_n, dist_temp


def dist_seq_rf(FEAT_SEQ, ng, nf, num_g, num_f):
    FEAT_A = []
    FEAT_P = []
    for idx, feat_seq in enumerate(FEAT_SEQ):
        feat_a = feat_seq[0:ng]
        feat_p = feat_seq[(ng+nf):(num_g+nf)]
        FEAT_A.append(feat_a)
        FEAT_P.append(feat_p)
    del FEAT_SEQ
    
    print("Calculating DTW distance... (Parallel CPU RF)")
    num_cores = max(1, multiprocessing.cpu_count() // 6) # Divide by 5-6 concurrent seeds to avoid RAM OOM
    results = Parallel(n_jobs=num_cores)(delayed(process_rf_user)(idx, feat_a, FEAT_P[idx], ng, FEAT_A, FEAT_P) for idx, feat_a in enumerate(FEAT_A))
    
    DIST_P = [r[0] for r in results]
    DIST_N = [r[1] for r in results]
    DIST_TEMP = [r[2] for r in results]
    
    DIST_P = numpy.concatenate(DIST_P, axis=0)
    DIST_N = numpy.concatenate(DIST_N, axis=0)
    DIST_TEMP = numpy.concatenate(DIST_TEMP, axis=0)
    return DIST_P, DIST_N, DIST_TEMP
