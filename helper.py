import numpy as np
from numpy import linalg as LA
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.optim import SGD
import math



def hello():
    print("hello!")


## Code to generate data for RPCA problem. References:
## https://github.com/caesarcai/LRPCA/blob/main/synthetic_data_exp/training_codes.py 
## https://github.com/caesarcai/AccAltProj_for_RPCA/blob/master/test_AccAltProj.m
## returns: L, S, M
def generate_problem(r,d1,d2,alpha, c):
    U0_t 		= torch.randn(d1,r)
    V0_t 		= torch.randn(d2,r)
    L0_t 		= U0_t @ V0_t.t()
    idx 		= torch.randperm(d1*d2)
    idx 		= idx[:math.floor(alpha * d1*d2)]
    s_range		= c * torch.mean(torch.abs(L0_t))
    S0_tmp 		= torch.rand(d1 * d2)
    S0_tmp 		= s_range * (2.0 * S0_tmp - 1.0)
    S0_t        = torch.zeros(d1 * d2)
    S0_t[idx]   = S0_tmp[idx]
    S0_t        = S0_t.reshape((d1,d2))
    Y0_t        = L0_t + S0_t
    return L0_t, S0_t, Y0_t



def thres(inputs, threshold, hard=True):
    if not hard:
        out = torch.sign(inputs) * torch.max(torch.abs(inputs)- threshold, torch.zeros([1,1]))
    else:
        out = inputs * (torch.abs(inputs) > threshold)
    return out



def resample(m, n, siz_row, siz_col):
    rows = np.random.randint(0, m, size=[1, siz_row])
    cols = np.random.randint(0, n, size=[1, siz_col])
    return [np.unique(rows), np.unique(cols)]



## returns: loss, L, S
def AccAltProj(M0, r, tol, gamma, max_iter):
    m, n = M0.shape
    norm_of_M0 = torch.linalg.norm(M0)
    ## Keep track of loss
    loss = []
    ## Initialization
    beta = 1/(2 * np.power(m * n, 1/4))
    beta_init = 4 * beta
    zeta = beta_init * torch.linalg.norm(M0, 2)
    S = thres(M0, zeta)#, False)
    U, Sigma, V = torch.linalg.svd(M0 - S, full_matrices=False)
    ## NOTE: torch.linalg.svd(M) returns U, S, V such that M=USV not USV.T
    U, Sigma, V = U[:,:r], Sigma[:r], V.t()[:, :r]
    L = U @ torch.diag(Sigma) @ V.t()
    zeta = beta * Sigma[0]
    S = thres(M0 - L, zeta)#, False)
    ## Initial loss
    err = torch.linalg.norm(M0 - L - S)/ norm_of_M0
    loss.append(err)
    for t in range(max_iter):
        ## Update L
        Z = M0 - S
        Q1, R1 = torch.linalg.qr(Z.t() @ U - V @ ((Z @ V).t() @ U)) ## reduced QR
        Q2, R2 = torch.linalg.qr(Z @ V - U @ (U.t() @ Z @ V)) ## reduced QR
        A = torch.cat((torch.cat((U.t() @ Z @ V, R1.t()), 1), 
                        torch.cat((R2, torch.zeros(R2.shape)), 1)), 0) ## A is 2r x 2r matrix
        Um, Sigma, Vm = torch.linalg.svd(A, full_matrices=False)
        U = torch.cat((U, Q2), 1) @ Um[:,:r]
        V = torch.cat((V, Q1), 1) @ Vm.t()[:,:r]
        L = U @ torch.diag(Sigma[:r]) @ V.t()
        ## Update S
        zeta = beta * (Sigma[r] + torch.pow(gamma, t + 1) * Sigma[0])
        S = thres(M0 - L, zeta)#, False)
        ## Compute error
        err = torch.linalg.norm(M0 - L - S)/ norm_of_M0
        loss.append(err)
        if err < tol:
            return loss, L, S
    return loss, L, S



## returns: loss, L, S
def IRCUR(M0, r, tol, gamma, con, max_iter):
    m, n = M0.shape
    norm_of_M0 = torch.linalg.norm(M0)
    ## Keep track of loss
    loss = []
    ## Initialization
    siz_row, siz_col = np.ceil(con * r * np.log(m)).astype(np.int64), np.ceil(con * r * np.log(n)).astype(np.int64)
    zeta = torch.max(torch.abs(M0))
    C, pinv_U, R = torch.zeros(M0.shape), torch.zeros([M0.shape[1], M0.shape[0]]), torch.zeros(M0.shape)
    for t in range(max_iter):
        ## resample rows and columns
        rows, cols = resample(m, n, siz_row, siz_col)
        M0_rows = M0[rows, :]
        M0_cols = M0[:, cols]
        norm_of_M0 = torch.linalg.norm(M0_rows) + torch.linalg.norm(M0_cols)
        ## compute submatrices of L from previous iteration
        L_rows = C[rows, :] @ pinv_U @ R
        L_cols = C @ pinv_U @ R[:, cols]
        ## update S using submatrices of L
        S_rows = thres(M0_rows - L_rows, zeta)
        S_cols = thres(M0_cols - L_cols, zeta)
        ## update L
        C = M0_cols - S_cols
        R = M0_rows - S_rows
        MU = C[rows, :]
        U,Sigma,Vh = torch.linalg.svd(MU, full_matrices=False)
        ## calculate Moore-Penrose inverse of Sigma
        pinv_U = Vh.t()[:,:r] @ torch.diag(1./Sigma[:r]) @ U[:, :r].t()
        ## update zeta
        zeta = gamma * zeta
        ## update loss
        err = (torch.linalg.norm(M0_rows - L_rows - S_rows) + torch.linalg.norm(M0_cols - L_cols - S_cols))/ norm_of_M0
        loss.append(err)
        if err < tol:
            L = C @ pinv_U @ R
            S = thres(M0 - L, zeta)
            return loss, L, S
    L = C @ pinv_U @ R
    S = thres(M0 - L, zeta)
    return loss, L, S



## generate train-test data, with 90-10 split
def generate_train_test(dataset_size, r,d1,d2,alpha, c):
    np.random.seed(0)
    torch.manual_seed(0)
    test = []
    train = []
    test_indices = random.sample(range(dataset_size), int(0.1 * dataset_size))
    for i in range(dataset_size):
        L, S, M = generate_problem(r,d1,d2,alpha, c)
        if i in test_indices:
            test.append((L, S, M))
        else:
            train.append((L, S, M))
    print(f"train dataset size: {len(train)}")
    print(f"test dataset size: {len(test)}")
    return train, test
    
    

def get_net_outputs(net_trained, net_bftrain, r, dataset):
    out_bftrain = []
    out_hat = []
    for L_true, S_true, M_true in dataset:
        L_bftrain, S_bftrain = net_bftrain(M_true, r)
        out_bftrain.append((L_bftrain, S_bftrain))
        plt.plot(np.array(net_bftrain.loss), '-b')
        
        L_hat, S_hat = net_trained(M_true, r)
        out_hat.append((L_hat, S_hat))
        plt.plot(np.array(net_trained.loss), 'r:')
    plt.show()
    return out_bftrain, out_hat



def plot_true_vs_est_matrices(L_hat, L_true, S_hat, S_true):
    combined = torch.cat((L_hat, L_true, S_hat, S_true))
    fig, [ax1, ax2] = plt.subplots(1,2)
    im1 = ax1.imshow(L_hat, vmin=torch.min(combined), vmax = torch.max(combined))
    ax1.set_title("L_hat")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(L_true, vmin=torch.min(combined), vmax = torch.max(combined))
    ax2.set_title("L_true")
    plt.colorbar(im2, ax=ax2)
    plt.show()
    fig, [ax1, ax2] = plt.subplots(1,2)
    im1 = ax1.imshow(S_hat, vmin=torch.min(combined), vmax = torch.max(combined))
    ax1.set_title("S_hat")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(S_true, vmin=torch.min(combined), vmax = torch.max(combined))
    ax2.set_title("S_true")
    plt.colorbar(im2, ax=ax2)
    plt.show()



def get_metrics(L_hat, L_true, S_hat, S_true):
    print("Before train")
    print("||L - L_hat||F:", torch.linalg.norm(L_true - L_bftrain)/ torch.linalg.norm(L_true))
    print("||S - S_hat||1:", torch.linalg.norm(S_true - S_bftrain, 1)/ torch.linalg.norm(S_true, 1))
    print("After train")
    print("||L - L_hat||F:", torch.linalg.norm(L_true - L_hat)/ torch.linalg.norm(L_true))
    print("||S - S_hat||1:", torch.linalg.norm(S_true - S_hat, 1)/ torch.linalg.norm(S_true, 1))
    print(torch.count_nonzero(S_true - S_bftrain))
    print(torch.count_nonzero(S_true - S_hat))
    
    

def get_metrics(true, out_classical, out_bftrain, out_hat):
    # between true and classical
    # between true and hat


