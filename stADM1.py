import numpy as np
import pandas as pd
import scanpy as sc

#读取数据

def svt(A, tau):
    #  矩阵初始化，生成一个和矩阵A形状一样的0矩阵
    A = A.astype(float)
    Y = np.zeros_like(A)
    shape = A.shape
    Omega = np.zeros(shape)
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if A[i, j] > 0:
                Omega[i, j] = 1
    epslion = 1e-2
    delta = None
    max_iterations = 100

    if not tau:
        tau = 5 * np.sum(A.shape) / 2
    if not delta:
        #  确定步长初始值
        delta = 1.2 * np.prod(A.shape) / np.sum(Omega)

    for _ in range(max_iterations):
        #  对矩阵Y进行奇异值分解
        U, S, V = np.linalg.svd(Y, full_matrices=False)
        #  soft-thresholding operator
        k = sum(S > tau)
        S = np.maximum(S - tau, 0)
        #  singular value shrinkage
        A_svt = np.linalg.multi_dot([U, np.diag(S), V])
        #  Y的迭代
        Y += delta * Omega * (A-A_svt)
        #  误差计算
        rel_recon_error = np.linalg.norm(Omega * (A_svt-A)) / np.linalg.norm(Omega*A)
        if rel_recon_error < epslion:
            break
    return A_svt,k



def ADMM(y, C_init, S_init,L, a1, a2, a3,maxiter=10,Lambda_init=None,
         abstol=1e-3, reltol=1e-3, rho=1, overrelax=1.5, candidate=0.05):
    # minimize 1/2||Y - (Z - S)||_F^2 + a1 *||C||_* + a2 * ||S||_1 + a3 * tr(C.T*L*C)
    # suject to Z = C, Z >= 0, P_Omega(S) = 0, P_{Omega^c}(S) >= 0

    # initialization
    n = y.shape[0]
    p = y.shape[1]
    Omega = y > candidate
    if (S_init is None):
        S_init = C_init * ~Omega + np.zeros((n, p)) * Omega
    if (Lambda_init is None):
        Lambda_init = np.zeros((n, p))
    if (a1 is None):
        a1 = (np.sqrt(n) + np.sqrt(p)) * np.std(y)
    if (a2 is None):
        a2 = np.std(y * Omega - C_init * Omega)
    C = C_init
    S = S_init
    Z = C;
    Lambda=Lambda_init
    alpha = overrelax


    # solve
    s_norm = []
    r_norm = []
    tol_pri = []
    tol_dual = []
    history = []
    for k in range( maxiter):
        print(k)

        # update Z,S
        # on Omega
        S1 = S * ~Omega
        tmp = np.linalg.inv(1 + rho + a3 * L + a3 * L.T) @ (y * Omega + rho * (C * Omega ) - Lambda * Omega)   #输出matrix
        Z1 = Z * ~Omega + np.maximum(tmp, 0) * Omega
        # on !Omega
        index = np.linalg.inv(1 + rho + L + L.T) @ (rho * (C * ~Omega) - (Lambda * ~Omega) - a2) < ((y * ~Omega) + a2)
        tmp1 = np.maximum( np.linalg.inv (1 + rho + L + L.T) @ ( y * ~Omega + rho * (C * ~Omega) - (Lambda * ~Omega) )  , 0)  # s = 0
        tmp2 = np.maximum( np.linalg.inv ( rho + L + L.T) @ ( rho * (C * ~Omega) - Lambda * ~Omega - a2 ), 0)  # s = z - a2 -y
        tmpS = np.maximum( tmp2 - y * ~Omega - a2 , 0)
        #tmpS = np.maximum( tmp2 - y * ~Omega , 0)
        tmpS = tmpS * ~index
        tmpZ = tmp1 * index + tmp2 * ~index
        S2 = S * Omega + tmpS * ~Omega
        Z2 = Z * Omega + tmpZ * ~Omega

        S = S1 * Omega + S2 * ~Omega
        Z = Z1 * Omega + Z2 * ~Omega

        # overrelaxation
        #Z_hat = alpha * Z + (1 - alpha) * C
        Z_hat = 0.5 * Z + 0.5 * C

        # update C
        C_old = C
        tmp = Z_hat + Lambda / rho
        svts=svt(tmp, a1 / rho)
        C=svts[0]
        C=np.maximum(C,0)
        r=svts[1]

        # update Lambda
        Lambda = Lambda + rho * (Z_hat - C)

        # diagnostics
        history.append(rho)
        history.append(np.linalg.norm(Z - C, 'fro'))
        history.append(np.linalg.norm((C - C_old), 'fro') * rho)
        history.append(np.sqrt(n * p) * abstol + reltol * np.maximum(np.linalg.norm(Z, 'fro'), np.linalg.norm(C,'fro')))
        history.append(np.sqrt(p * n) * abstol + reltol * np.linalg.norm(Lambda, 'fro'))

        if (history[2] > 10 * history[1]):
            rho = rho * 2
        elif (history[1] > 10 * history[2]):
            rho = np.maximum(rho / 2, 1e-4)

    exprs = y
    exprs[S > 0] = C[S > 0]
    exprs[exprs < 0] = 0
    return C, S, r, a1, exprs, a2, history



C, S, r, a1, exprs, a2, history = ADMM(X, c,None,l,a1,a2,a3,1,None,1e-3, 1e-3, 1, 1.5, 0.05)

S_range = S > 0
C_end = C * S_range
Y = C_end + X
print("结果C_end：\n", C_end)
print("结果X：\n", X)
print("结果Y：\n", Y)







