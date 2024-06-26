import torch
import numpy as np
import math
from depth_cov.utils.utils import safe_sqrt
from depth_cov.utils import lin_alg
from scipy.special import kv, gamma
from scipy.special import hermite

# from typing import float 

def quadratic(x, A):
  x_sq = torch.square(x)
  x_corr = x[...,0]*x[...,1]
  xtAx = A[...,0,0]*x_sq[...,0] + 2*A[...,0,1]*x_corr + A[...,1,1]*x_sq[...,1]
  return xtAx

# https://etd.ohiolink.edu/apexprod/rws_etd/send_file/send?accession=osu1437409380&disposition=attachment
def nonstationary(x1, E1, x2, E2):
  diff = (x1.unsqueeze(2) - x2.unsqueeze(1)).float()
  E_sum = E1.unsqueeze(2) + E2.unsqueeze(1)
  E_sum_inv, E_sum_det = lin_alg.inv2x2(E_sum)
  Q = 0.5 * quadratic(diff, E_sum_inv)

  E1_det_quarter_root = torch.sqrt(torch.sqrt(torch.linalg.det(E1)))
  E2_det_quarter_root = torch.sqrt(torch.sqrt(torch.linalg.det(E2)))
  C = 2.0 * E1_det_quarter_root.unsqueeze(2) * E2_det_quarter_root.unsqueeze(1) / torch.sqrt(E_sum_det)

  return Q, C


# https://www.jmlr.org/papers/volume5/jebara04a/jebara04a.pdf
# bhattacharyya kernel: p=0.5 gives K(x,x) = 1
# expected likelihood kernel: p=1.0
# Assumess D=2, p=0.5
def prob_product_quad(x1, E1, x2, E2):
    num = 10  # Hyperparameter. Approximation with degree n
    a = 10  # Hyperparameter
    n = 1 / (10 * np.sqrt(2))  # l = 10
    b = (1 + (2 * n / a) ** 2) ** (0.25)
    d = (a / 2) * np.sqrt(b ** 2 - 1)
    L = np.zeros(num + 1)
    
    for i in range(1, num + 1):
        L[i] = np.sqrt((a * a) / (a * a + d * d + n * n)) * ((n * n / (a * a + d * d + n * n)) ** i)
    
    K = bhattacharyya_kernel(x1, E1, x2, E2)
    res = torch.zeros(K.shape)

    for i in range(1, num + 1):
        H_i = hermite(i)
        pi1_i = np.sqrt(b / np.math.factorial(i)) * torch.exp(-a * a * x1.unsqueeze(-1) * x1.unsqueeze(-1)) * \
                torch.from_numpy(H_i(np.sqrt(2) * a * b * x1.unsqueeze(-1)).astype(np.float32))
        pi2_i = np.sqrt(b / np.math.factorial(i)) * torch.exp(-a * a * x2.unsqueeze(-2) * x2.unsqueeze(-2)) * \
                torch.from_numpy(H_i(np.sqrt(2) * a * b * x2.unsqueeze(-2)).astype(np.float32))

        res += L[i] * (pi1_i * pi2_i * K)

    return res

def bhattacharyya_kernel(x1, E1, x2, E2):
    diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).float()

    Q = (E1[...,1,1].unsqueeze(-2) + E2[...,1,1].unsqueeze(-3)) * torch.square(diff[...,0])
    Q += -2 * (E1[...,0,1].unsqueeze(-2) + E2[...,0,1].unsqueeze(-3)) * diff[...,0] * diff[...,1]
    Q += (E1[...,0,0].unsqueeze(-2) + E2[...,0,0].unsqueeze(-3)) * torch.square(diff[...,1])

    denominator = (E1[...,0,0].unsqueeze(-2) + E2[...,0,0].unsqueeze(-3)) * (E1[...,1,1].unsqueeze(-2) + E2[...,1,1].unsqueeze(-3)) \
                  - torch.square(E1[...,0,1].unsqueeze(-2) + E2[...,0,1].unsqueeze(-3))

    Q /= denominator
    Q *= 0.5

    return torch.exp(-Q)

# Assumes D=2, p=0.5
def prob_product_constant(E1, E2):
  dim1 = len(E1.shape)-2
  dim2 = len(E2.shape)-3

  E1_det_root = lin_alg.det2x2(E1) ** (0.25)
  E2_det_root = lin_alg.det2x2(E2) ** (0.25)
  C = 2.0 * E1_det_root.unsqueeze(dim1) * E2_det_root.unsqueeze(dim2) / safe_sqrt((E1[...,0,0].unsqueeze(dim1) + E2[...,0,0].unsqueeze(dim2)) * (E1[...,1,1].unsqueeze(dim1) + E2[...,1,1].unsqueeze(dim2)) - torch.square(E1[...,0,1].unsqueeze(dim1) + E2[...,0,1].unsqueeze(dim2)))  

  return C

## Diagonal covariance functions

def diagonal_nonstationary(coords, E):
  K_diag = torch.ones(coords.shape[0], coords.shape[1], device=coords.device)
  return K_diag

def diagonal_prob_product(coords, E):
  E_det_root = torch.sqrt(lin_alg.det2x2(E))
  E_sum_det = lin_alg.det2x2(2*E)
  C = 2.0 * E_det_root / safe_sqrt(E_sum_det)
  Q = torch.zeros_like(C)
  return Q, C

## Isotropic covariance functions
def squared_exponential(Q):
  K = torch.exp(-0.5*Q)
  return K

def matern(Q):
  Q_sqrt = safe_sqrt(Q) # Constant term for stability, otherwise nan on backward
  # v=3/2
  tmp = (np.sqrt(3))*Q_sqrt
  k_v_3_2 = (1 + tmp) * torch.exp(-tmp)

  K = k_v_3_2
  return K

# Construct convolution of heteregenous Gaussian kernels on R2 by Chris Paciorek thesis
def convolutionOfGaussianCovariance(x1, E1, x2, E2):

    diff = (x1.unsqueeze(2) - x2.unsqueeze(1)).float()

    E_sum = E1.unsqueeze(2) + E2.unsqueeze(1)
    E_sum_inv, E_sum_det = lin_alg.inv2x2(E_sum)

    diff_sq = torch.square(diff)
    diff_corr = diff[...,0]*diff[...,1]
    Q = 0.5 * (E_sum_inv[...,0,0]*diff_sq[...,0] + 2*E_sum_inv[...,0,1]*diff_corr + E_sum_inv[...,1,1]*diff_sq[...,1])

    k = x1.shape[-1]
    C = 1.0/torch.sqrt( ((2*np.pi)**k) * E_sum_det)
    K = C * torch.exp(-Q)

    return K

def gaussianKernel(x1, E1, x2):

    diff = (x1.unsqueeze(2) - x2.unsqueeze(1)).float()
    E1_inv, E1_det = lin_alg.inv2x2(E1)

    diff_sq = torch.square(diff)
    diff_corr = diff[...,0]*diff[...,1]
    Q = 0.5*(E1_inv[...,0,0].unsqueeze(-1)*diff_sq[...,0] + 2*E1_inv[...,0,1].unsqueeze(-1)*diff_corr + E1_inv[...,1,1].unsqueeze(-1)*diff_sq[...,1])
        
    k = x1.shape[-1]
    C = 1.0/safe_sqrt( ((2*np.pi)**k) * E1_det)
    K = C.unsqueeze(2) * torch.exp(-Q)

    return K

def diagonalCorrelation(E):
    C = torch.ones(E.shape[0], E.shape[1], device=E.device)
    return C

def diagonalConvolutionOfGaussianCovariance(E):
    E_det = lin_alg.det2x2(E)
    k = E.shape[-1]
    C = 0.5/safe_sqrt( ((2*np.pi)**k) * E_det)
    return C

def diagonalGaussianKernel(E):
    E_det = lin_alg.det2x2(E)
    k = E.shape[-1]
    C = 1.0/safe_sqrt( ((2*np.pi)**k) * E_det)
    return C

# NEW Kernel
def new_kernel(x1, E1, x2, E2):
    d = np.linalg.norm(x1 - x2)
    
    # Parameters for the Matern kernel with nu=1.5
    nu = 1.5
    sqrt_2nu = np.sqrt(2 * nu)
    l = 5
    # Calculate the Matern kernel
    coefficient = 1 / (gamma(nu) * 2**(nu - 1))
    scaled_distance = (sqrt_2nu / l) * d
    bessel_term = kv(nu, scaled_distance)
    
    kernel_value = coefficient * (scaled_distance**nu) * bessel_term
    
    return kernel_value

# FAMGP approximated kernel
def FAMGP_kernel(x1, x2):
  num = 10 # Hyperparameter. Approximation with degree n
  l = 10 # Hyperparameter.
  a = 10 # Hyperparameter
  res = 0
  L = [0] * (num + 1)
  pi1 = [0] * (num + 1)
  pi2 = [0] * (num + 1)
  n = 1 / (l * np.sqrt(2))
  b = (1 + (2*n/a) ** 2) ** (0.25)
  d = (a / 2) * np.sqrt(b ** 2 - 1)
  for i in range (1..num):
    L[i] = np.sqrt((a * a)/(a * a + d * d + n * n)) * ((n*n/(a*a + d*d + n*n)) ** i)
  for i in range (1..num):
    H = hermite(i)
    pi1[i] = np.sqrt(b/math.factorial(i))*torch.exp(-a*a*x1*x1)*H(np.sqrt(2)*a*b*x1)
  for i in range (1..num):
    H = hermite(i)
    pi2[i] = np.sqrt(b/math.factorial(i))*torch.exp(-a*a*x1*x1)*H(np.sqrt(2)*a*b*x1)
  for i in range (1..num):
     res += L[i] * pi1[i] * pi2[i]
  return res