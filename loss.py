import torch
import torch.nn as nn
import numpy as np

PI = torch.tensor(np.pi)
PI2 = torch.tensor(np.pi*2)
LOG2 = torch.tensor(2.0).log()

def masked_mse_loss(input, target, mask):
    loss = torch.sum(((input - target) ** 2) * mask)
    num_valid_kps = torch.sum(mask).item()
    if num_valid_kps != 0:
        loss /= num_valid_kps
    return loss

def masked_l1_loss(input, target, mask):
    loss = torch.sum(torch.abs(input-target) * mask)
    num_valid_kps = torch.sum(mask).item()
    if num_valid_kps != 0:
        loss /= num_valid_kps
    return loss

def masked_nll_gaussian(input, target, rho, mask, beta=1.0, computing_mode=0, offset1=1, offset2=0.1):
    '''
    masked negative log likelihood gaussian
    :param input: B x N x M
    :param target: B x N x M
    :param rho: B x N x M
    :param mask: B x N x 1 (or B x N x M)
    :return:
    '''

    diff = (input - target) ** 2

    if computing_mode == 0:
        # Method 1, rho = log(variance)
        # weight = torch.exp(-rho)
        # nll_gauss = 0.5 * (weight * diff + logvar )
        # Method 2, rho = log(2PI*variance)
        weight = torch.exp(-rho)
        nll_gauss = PI * weight * diff + 0.5*beta*rho
    elif computing_mode == 1:
        # Method 3, sigma = alpha * sigmoid(rho) + beta
        sigma = offset1 * torch.sigmoid(rho) + offset2
        variance = sigma * sigma
        nll_gauss = diff / (2*variance) + beta * 0.5 * torch.log(2*PI*variance)
    elif computing_mode == 2:
        # rho = log(variance), but the regularizer is |sigma - 1|**2
        sigma = torch.exp(rho / 2)
        variance = sigma ** 2
        nll_gauss =  diff / variance + beta * (sigma - 1)**2
    elif computing_mode == 3:
        # rho = log(variance), but the regularizer is sigma**2
        sigma = torch.exp(rho / 2)
        variance = sigma ** 2
        nll_gauss = diff / variance + beta * variance
    elif computing_mode == 4:
        # rho = log(variance), but the regularizer is |1/sigma - 1|**2
        sigma = torch.exp(rho / 2)
        variance = sigma ** 2
        nll_gauss = diff / variance + beta * (1/sigma - 1) ** 2


    loss = torch.sum(nll_gauss * mask)
    num_valid_kps = torch.sum(mask).item()
    if num_valid_kps != 0:
        loss /= num_valid_kps
    # print('w: ', weight, 'diff: ', torch.sum(diff * mask).cpu().detach().numpy(), 'logvar: ', logvar.cpu().detach().numpy())
    # if loss < 0:
    #     print('lo: ', loss.cpu().detach().numpy())
    return loss

def masked_nll_gaussian2(input, target, rho, patches_rho, mask, beta=1.0, gamma=1.0):
    '''
    masked negative log likelihood gaussian
    :param input: B x N x M
    :param target: B x N x M
    :param rho: B x N x M
    :param mask: B x N x 1 (or B x N x M)
    :return:
    '''

    diff = (input - target) ** 2

    # Method 1, rho = log(variance), patches_rho = log(w)
    variance = torch.exp(rho)
    w = torch.exp(patches_rho)
    nll_gauss = 0.5 * diff / (variance + w + 1e-6) + 0.5 * beta * torch.log(variance + gamma*w + 1e-6) + 0.5*torch.log(2*PI)


    loss = torch.sum(nll_gauss * mask)
    num_valid_kps = torch.sum(mask).item()
    if num_valid_kps != 0:
        loss /= num_valid_kps
    # print('w: ', weight, 'diff: ', torch.sum(diff * mask).cpu().detach().numpy(), 'logvar: ', logvar.cpu().detach().numpy())
    # if loss < 0:
    #     print('lo: ', loss.cpu().detach().numpy())
    return loss

def masked_nll_gaussian_covar(input, target, rho, kp_mask, patches_rho=None, loss_fun_mode=0, penalty_mode=0, beta=1.0, beta_loc_uc=1.0, offset1=6, offset2=0.5):
    '''
    masked negative log likelihood gaussian; covariance for single kps
    :param input: B x N x 2
    :param target:B x N x 2
    :param rho: B x N x (2*d)
    :param kp_mask: B x N
    :param patches_rho: N (only use support patches to compute weight inv w) or B x N (use both support & query patches)

    :return:
    '''
    B, N, _= input.shape
    d = rho.shape[2] // 2  # covariance latent Q: 2 x d
    Q = rho.reshape(B, N, 2, d)
    Omega = torch.zeros(B, N, 2, 2, requires_grad=True).cuda()
    for i in range(B):
        for j in range(N):
            q = Q[i, j]
            Omega[i, j] = q.matmul(q.permute(1, 0))
    Omega = Omega / d  # B x N x 2 x 2, precision matrix inv(Covar)
    mask_ones = torch.diag(torch.ones(2)).cuda()

    if patches_rho is not None:  # used for multiple kps' covar
        dim_num_of_patches_rho = len(patches_rho.shape)
    diff = (input - target)  # B x N x 2
    loss = 0
    # vlist = []
    for i in range(B):
        for j in range(N):
            if kp_mask[i,j] == 0:
                continue
            v = diff[i,j]
            precision_matrix = Omega[i, j]

            if loss_fun_mode == 0:  # (x-u)^t * [Omega + beta*W^-1] * (x-u) - log(det(Omega)) - beta*log(det(W^-1))
                # nll_gauss = 0.5*v.matmul(precision_matrix).matmul(v) - 0.5 * torch.log(torch.det(precision_matrix) ) # + 1e-6) # + torch.log(PI2)  # slow
                # det_precision_matrix = torch.det(precision_matrix)  # slow
                det_precision_matrix = precision_matrix[0, 0] * precision_matrix[1, 1] - precision_matrix[0, 1]*precision_matrix[1, 0]
                # print(0.5*v.matmul(precision_matrix).matmul(v))
                # print(0.5 * torch.log(det_precision_matrix))
                # print(torch.inverse(precision_matrix))
                # if det_precision_matrix == 0:  # we are very careful to the zero
                #     det_precision_matrix +=  1e-6
                nll_gauss = 0.5*v.matmul(precision_matrix).matmul(v) - 0.5 * torch.log(det_precision_matrix+1e-9)
                # nll_gauss = 0.5 * v.matmul(precision_matrix).matmul(v) - 0.5 * torch.log(precision_matrix[0, 0] * precision_matrix[1, 1] - precision_matrix[0, 1]*precision_matrix[1, 0])
                nll_gauss *= beta_loc_uc  # added to control loc uc


                # -----------------------------
                # take the weight which is learned from patches into consideration
                if patches_rho is not None:
                    if dim_num_of_patches_rho == 1:  # only use support patches to compute inv_w, size is T
                        inv_w = patches_rho[j]
                    elif dim_num_of_patches_rho == 2: # use both support & query patches, size is B2 x T
                        inv_w = patches_rho[i, j]
                    if penalty_mode == 0:
                        nll_gauss_w = 0.5*(torch.sum(v**2 * inv_w) - 2*torch.log(inv_w+1e-9))  # -torch.log(inv_w[j] ** 2)
                    elif penalty_mode == 1:
                        nll_gauss_w = torch.sum(v**2 * inv_w) + (inv_w - 1) ** 2
                    elif penalty_mode == 2:
                        nll_gauss_w = torch.sum(v ** 2 * inv_w) + (1 / inv_w - 1) ** 2
                    if beta == 0:
                        exit('Error as beta cannot be 0. The offset regression not using beta causes conflit as grid-cls uses it (sqrt(w)). Just turn off use_pum.')
                    else:
                        nll_gauss += beta*nll_gauss_w
                # -----------------------------
            elif loss_fun_mode == 1:  # (x-u)^t * [Omega + beta*W^-1] * (x-u) - log(det(Omega + beta*W^-1))
                if dim_num_of_patches_rho == 1:  # only use support patches to compute inv_w, size is T
                    inv_w = patches_rho[j]
                elif dim_num_of_patches_rho == 2:  # use both support & query patches, size is B2 x T
                    inv_w = patches_rho[i, j]
                inv_w = torch.diag(inv_w.repeat(2))
                precision_matrix = precision_matrix + beta * inv_w
                det_precision_matrix = precision_matrix[0, 0] * precision_matrix[1, 1] - precision_matrix[0, 1] * precision_matrix[1, 0]
                if det_precision_matrix <= 1e-8:  # we are very careful to the zero
                    det_precision_matrix +=  1e-6
                nll_gauss = 0.5*v.matmul(precision_matrix).matmul(v) - 0.5 * torch.log(det_precision_matrix)
            elif loss_fun_mode == 2:  # (x-u)^t * [Covar + beta*W]^-1 * (x-u) + log(det(Covar + beta*W))
                if dim_num_of_patches_rho == 1:  # only use support patches to compute weight, size is T
                    w = patches_rho[j]
                elif dim_num_of_patches_rho == 2:  # use both support & query patches, size is B2 x T
                    w = patches_rho[i, j]
                w_matrix = torch.diag(w.repeat(2))
                covar_matrix = precision_matrix + beta * w_matrix
                inv_covar_matrix = torch.inverse(covar_matrix)
                det_covar_matrix = covar_matrix[0, 0] * covar_matrix[1, 1] - covar_matrix[0, 1]*covar_matrix[1, 0]
                if det_covar_matrix == 0:
                    det_covar_matrix += 1e-6
                nll_gauss = 0.5*v.matmul(inv_covar_matrix).matmul(v) + 0.5 * torch.log(det_covar_matrix)
            elif loss_fun_mode == 3:  # Omega' = Omega * W^-1, but only perform multiplication on the diagonal
                if dim_num_of_patches_rho == 1:  # only use support patches to compute inv_w, size is T
                    inv_w = patches_rho[j]
                elif dim_num_of_patches_rho == 2:  # use both support & query patches, size is B2 x T
                    inv_w = patches_rho[i, j]
                inv_w_matrix = torch.diag(inv_w.repeat(2))
                # mask_ones = torch.diag(torch.ones(2))
                precision_matrix2 = precision_matrix * mask_ones * inv_w_matrix + precision_matrix * (1-mask_ones)
                det_precision_matrix = precision_matrix[0, 0] * precision_matrix[1, 1] - precision_matrix[0, 1]*precision_matrix[1, 0]
                if det_precision_matrix <= 1e-8:  # we are very careful to the zero
                    det_precision_matrix +=  1e-6
                nll_gauss = 0.5*v.matmul(precision_matrix2).matmul(v) - 0.5 * torch.log(det_precision_matrix) + (1 / inv_w - 1) ** 2
                # if torch.isnan(nll_gauss):
                #     print(nll_gauss)
            elif loss_fun_mode == 4:  # Omega' = W^-0.5 * Omega * W^-0.5
                if dim_num_of_patches_rho == 1:  # only use support patches to compute inv_w, size is T
                    inv_w = patches_rho[j]
                elif dim_num_of_patches_rho == 2:  # use both support & query patches, size is B2 x T
                    inv_w = patches_rho[i, j]
                precision_matrix = precision_matrix * inv_w
                det_precision_matrix = precision_matrix[0, 0] * precision_matrix[1, 1] - precision_matrix[0, 1]*precision_matrix[1, 0]
                if det_precision_matrix == 0:  # we are very careful to the zero
                    det_precision_matrix +=  1e-6
                # det_precision_matrix *= (inv_w * inv_w)
                nll_gauss = 0.5*v.matmul(precision_matrix).matmul(v) - 0.5 * torch.log(det_precision_matrix)
            elif loss_fun_mode == 5:  # Omega' = Omega + (W^-0.5)^T*(W^-0.5)
                if dim_num_of_patches_rho == 1:  # only use support patches to compute inv_w, size is T
                    inv_w = patches_rho[j]
                elif dim_num_of_patches_rho == 2:  # use both support & query patches, size is B2 x T
                    inv_w = patches_rho[i, j]
                # precision_matrix = precision_matrix + beta*inv_w
                det_precision_matrix = precision_matrix[0, 0] * precision_matrix[1, 1] - precision_matrix[0, 1]*precision_matrix[1, 0]
                if det_precision_matrix == 0:  # we are very careful to the zero
                    det_precision_matrix +=  1e-6
                nll_gauss = 0.5 * v.matmul(precision_matrix).matmul(v) - 0.5 * torch.log(det_precision_matrix)
                nll_gauss_w = 0.5 * (inv_w * (torch.sum(v) ** 2) - torch.log(inv_w))
                nll_gauss += beta * nll_gauss_w

            loss += nll_gauss


            # vlist.append(nll_gauss)

            # if nll_gauss.item() < -100:
            #     print(precision_matrix)
            #     print(torch.det(precision_matrix))
            #     print(nll_gauss)
            #     exit(0)

    num_valid_kps = torch.sum(kp_mask).item()

    if num_valid_kps != 0:
        loss /= num_valid_kps
        # loss = sum(vlist) / num_valid_kps
    else:
        loss = torch.tensor(0.)

    return loss

def compute_determinant(m, order=2):
    if order == 2:
        det = m[0, 0] * m[1, 1] - m[0, 1]*m[1, 0]
    elif order == 3:
        det = m[0, 0] * m[1, 1] * m[2, 2] + m[0, 1] * m[1, 2] * m[2, 0] + m[0, 2] * m[1, 0] * m[2, 1] - \
                m[0, 2] * m[1, 1] * m[2, 0] - m[0, 1] * m[1, 0] * m[2, 2] - m[0, 0] * m[1, 2] * m[2, 1]
    elif order == 4:
        det = m[0, 0] * (m[1, 1] * m[2, 2] * m[3, 3] + m[1, 2] * m[2, 3] * m[3, 1] + m[1, 3] * m[2, 1] * m[3, 2] - \
                m[1, 3] * m[2, 2] * m[3, 1] - m[1, 2] * m[2, 1] * m[3, 3] - m[1, 1] * m[2, 3] * m[3, 2]) - \
              m[0, 1] * (m[1, 0] * m[2, 2] * m[3, 3] + m[1, 2] * m[2, 3] * m[3, 0] + m[1, 3] * m[2, 0] * m[3, 2] - \
                m[1, 3] * m[2, 2] * m[3, 0] - m[1, 2] * m[2, 0] * m[3, 3] - m[1, 0] * m[2, 3] * m[3, 2]) + \
              m[0, 2] * (m[1, 0] * m[2, 1] * m[3, 3] + m[1, 1] * m[2, 3] * m[3, 0] + m[1, 3] * m[2, 0] * m[3, 1] - \
                m[1, 3] * m[2, 1] * m[3, 0] - m[1, 1] * m[2, 0] * m[3, 3] - m[1, 0] * m[2, 3] * m[3, 1]) - \
              m[0, 3] * (m[1, 0] * m[2, 1] * m[3, 2] + m[1, 1] * m[2, 2] * m[3, 0] + m[1, 2] * m[2, 0] * m[3, 1] - \
                m[1, 2] * m[2, 1] * m[3, 0] - m[1, 1] * m[2, 0] * m[3, 2] - m[1, 0] * m[2, 2] * m[3, 1])
    else:  # others
        det = torch.det(m)

    return det

def masked_nll_gaussian_covar2(input, target, rho, kp_mask, covar_method=0, patches_rho=None, loss_fun_mode=0, penalty_mode=0,  beta=1.0, beta_loc_uc=1.0, offset1=6, offset2=0.5):
    '''
    masked negative log likelihood gaussian; covariance for multiple kps
    :param input: B x N x 2
    :param target:B x N x 2
    :param rho: B x N x N x (2d)
    :param kp_mask: B x N
    :param patches_rho: B x N

    :return:
    '''
    B, N, _= input.shape
    d = rho.shape[3] // 2  # each block: 2 x d
    Q = rho.reshape(B, N, N, 2, d)
    mask_ones = torch.diag(torch.ones(2 * N)).cuda()

    num_valid_kps = torch.sum(kp_mask).item()

    # set u-\hat{u} to be 0, don't make change to Q
    if covar_method == 0:
        Q = Q.permute(0, 1, 3, 2, 4).reshape(B, 2*N, d*N)
        # Omega = torch.zeros(B, 2*N, 2*N, requires_grad=True).cuda()
        # for i in range(B):
        #     q = Q[i]
        #     Omega[i] = q.matmul(q.permute(1, 0))
        # Omega = Omega / (d*N)  # B x (2*N) x (2*N), precision matrix inv(Covar)
        Omega = []
        for i in range(B):
            q = Q[i]
            m = q.matmul(q.permute(1, 0)).div(d*N)  # 2N x 2N
            # m = Q[i].matmul(Q[i].permute(1, 0)).div(d*N)
            Omega.append(m)

        # -----------------------------
        # take the weight which is learned from patches into consideration
        if patches_rho is not None:
            # expand inv_w of size N into size 2N, namely, inv_w' = [w1, w1, w2, w2, w3, w3, ..., w_N, w_N], since v is vector [x, y]^{T}
            inv_w_2N = patches_rho.reshape(B, N, 1).repeat(1, 1, 2).reshape(B, 2*N)  # B x 2N

        # -----------------------------

        diff = (input - target)  # B x N x 2
        diff = diff * kp_mask.view(B, N, 1)
        diff = diff.reshape(B, 2*N)
        loss = 0
        for i in range(B):
            v = diff[i]  # 2N
            m = Omega[i]  # 2N x 2N

            if loss_fun_mode == 0:  # (x-u)^t * [Omega + beta*W^-1] * (x-u) - log(det(Omega)) - beta*log(det(W^-1))
                if N == 1:  # covariance for 1 kp
                    det_precision_matrix = m[0, 0] * m[1, 1] - m[0, 1] * m[1,0]
                else:  # covariance for 2 kps, 4 x 4, or 3 kps, 6 x 6, or higher orders
                    det_precision_matrix = torch.det(m)
                # if det_precision_matrix == 0:
                #     det_precision_matrix += 1e-6
                nll_gauss = 0.5 * v.matmul(m).matmul(v) - 0.5 * torch.log(det_precision_matrix+1e-9)
                nll_gauss *= beta_loc_uc  # added to control loc uc

                # -----------------------------
                # take the weight which is learned from patches into consideration
                if patches_rho is not None:
                    inv_w_temp = inv_w_2N[i]
                    if penalty_mode == 0:
                        nll_gauss_w = 0.5*(torch.sum(v**2 * inv_w_temp) - torch.log(torch.prod(inv_w_temp)+1e-9))
                    elif penalty_mode == 1:
                        nll_gauss_w = torch.sum(v**2 * inv_w_temp) + torch.sum((inv_w_temp - 1) ** 2)
                    elif penalty_mode == 2:
                        nll_gauss_w = torch.sum(v ** 2 * inv_w_temp) + torch.sum((1 / inv_w_temp - 1) ** 2)
                    if beta == 0:
                        exit('Error as beta cannot be 0. The offset regression not using beta causes conflit as grid-cls uses it (sqrt(w)). Just turn off use_pum.')
                    else:
                        nll_gauss += beta*nll_gauss_w
                # -----------------------------
            elif loss_fun_mode == 1:  # (x-u)^t * [Omega + beta*W^-1] * (x-u) - log(det(Omega + beta*W^-1))
                inv_w_temp = inv_w_2N[i]
                inv_w_temp = torch.diag(inv_w_temp)
                precision_matrix = m + beta * inv_w_temp
                det_precision_matrix = torch.det(precision_matrix)
                if det_precision_matrix <= 1e-8:
                    det_precision_matrix += 1e-6
                nll_gauss = 0.5 * v.matmul(precision_matrix).matmul(v) - 0.5 * torch.log(det_precision_matrix)
            elif loss_fun_mode == 2:  # (x-u)^t * [Covar + beta*W]^-1 * (x-u) + log(det(Covar + beta*W))
                w_temp = inv_w_2N[i]
                w_temp = torch.diag(w_temp)
                covar_matrix = m + beta * w_temp
                inv_covar_matrix = torch.inverse(covar_matrix)
                det_covar_matrix = torch.det(covar_matrix)
                if det_covar_matrix == 0:
                    det_covar_matrix += 1e-6
                nll_gauss = 0.5 * v.matmul(inv_covar_matrix).matmul(v) + 0.5 * torch.log(det_covar_matrix)
            elif loss_fun_mode == 3:  # Omega' = Omega * W^-1, but only perform multiplication on the diagonal
                inv_w_temp = inv_w_2N[i]
                inv_w_matrix_temp = torch.diag(inv_w_temp)
                # mask_ones = torch.diag(torch.ones(2*N))
                precision_matrix2 = m * mask_ones * inv_w_matrix_temp + m * (1 - mask_ones)
                det_precision_matrix = torch.det(m)
                if det_precision_matrix <= 1e-8:
                    det_precision_matrix += 1e-6
                nll_gauss = 0.5 * v.matmul(precision_matrix2).matmul(v) - 0.5 * torch.log(det_precision_matrix) + torch.sum((1 / inv_w_temp - 1) ** 2)
            elif loss_fun_mode == 4:  # Omega' = W^-0.5 * Omega * W^-0.5
                inv_w_temp = inv_w_2N[i]
                half_inv_w_temp = torch.sqrt(inv_w_temp)
                half_inv_w_temp = torch.diag(half_inv_w_temp)
                precision_matrix = half_inv_w_temp.matmul(m).matmul(half_inv_w_temp)
                det_precision_matrix = torch.det(precision_matrix)
                if det_precision_matrix == 0:
                    det_precision_matrix += 1e-6
                nll_gauss = 0.5 * v.matmul(precision_matrix).matmul(v) - 0.5 * torch.log(det_precision_matrix)
            elif loss_fun_mode == 5:  # Omega' = Omega + (W^-0.5)^T*(W^-0.5)
                inv_w_temp = inv_w_2N[i]
                half_inv_w_temp = torch.sqrt(inv_w_temp).reshape(2*N, 1)
                # inv_w_matrix = half_inv_w_temp.matmul(half_inv_w_temp.t())
                # precision_matrix = m + beta*inv_w_matrix
                det_precision_matrix = torch.det(m)
                if det_precision_matrix == 0:
                    det_precision_matrix += 1e-6
                nll_gauss = 0.5 * v.matmul(m).matmul(v) - 0.5 * torch.log(det_precision_matrix)
                nll_gauss_w = 0.5 * (torch.sum(half_inv_w_temp * v) ** 2 - torch.log(torch.prod(inv_w_temp)))
                nll_gauss += beta * nll_gauss_w

            loss += nll_gauss
    # cross-out / remove the blocks in Q and only take the corresponding valid kps and blocks into compute
    elif covar_method == 1:
        loss = 0
        for i in range(B):
            q = Q[i]
            n = torch.sum(kp_mask[i]).long().item()  # for image i there are n valid kps
            if n == 0:
                continue
            A = torch.zeros(n, n, 2, d, requires_grad=True).cuda()
            valid_j_cnt, valid_k_cnt = 0, 0
            for j in range(N):
                if kp_mask[i, j] == 0:
                    continue
                valid_j_cnt += 1
                valid_k_cnt = 0
                for k in range(N):
                    if kp_mask[i, k] == 0:
                        continue
                    valid_k_cnt += 1
                    A[valid_j_cnt-1, valid_k_cnt-1] = q[j, k]
            A = A.permute(0, 2, 1, 3).reshape(2*n, d*n)
            Omega = A.matmul(A.permute(1, 0)) # 2n x 2n
            Omega = Omega / (d * n)  # B x (2*n) x (2*n), precision matrix inv(Covar)
            index =(kp_mask[i]).bool()
            diff = input[i] - target[i]
            v = diff[index]
            v = v.reshape(2*n)
            if n == 1:  # covariance for 1 kp
                det_precision_matrix = Omega[0, 0] * Omega[1, 1] - Omega[0, 1] * Omega[1,0]
            else:  # covariance for 2 kps, 4 x 4, or 3 kps, 6 x 6, or higher orders
                det_precision_matrix = torch.det(Omega)
            if det_precision_matrix == 0:
                det_precision_matrix += 1e-6
            nll_gauss = 0.5 * v.matmul(Omega).matmul(v) - 0.5 * torch.log(det_precision_matrix)
            loss += nll_gauss / n

        if num_valid_kps != 0:
            loss /= B
        else:
            loss = torch.tensor(0.)
        return loss


    if num_valid_kps != 0:
        loss /= num_valid_kps
    else:
        loss = torch.tensor(0.)

    return loss


def masked_nll_laplacian(input, target, rho, mask, beta=1.0, computing_mode=0, offset1=1, offset2=0.1):
    '''
    masked negative log likelihood Laplacian
    :param input: B x N x 2
    :param target: B x N x 2
    :param rho: B x N x 2
    :param mask: B x N x 1 (or B x N x 2)
    :return:
    '''

    diff = torch.abs(input - target)

    if computing_mode == 0:
        # Method 1, rho = log(2b)
        # weight = torch.exp(-rho)
        # nll_laplacian = 2 * weight * diff + beta*rho
        # ---
        # Method 1, rho = log(b)
        weight = torch.exp(-rho)
        # nll_laplacian = weight * diff + beta*(rho + LOG2)
        nll_laplacian = weight * diff + beta * (rho)
    elif computing_mode == 1:  # offset1 and offset2 used only when computing_mode = 1
        # Method 2, b = alpha * sigmoid(logvar) + beta
        b = offset1 * torch.sigmoid(rho) + offset2
        # nll_laplacian = diff / b + beta * torch.log(2*b)
        nll_laplacian = diff / b + beta * torch.log(b)
    elif computing_mode == 2:
        # Method 3, rho = log(2b), but the regularizer is |b -1|
        b = torch.exp(rho) / 2
        nll_laplacian = diff / b + beta * torch.abs(b - 1)
    elif computing_mode == 3:
        # Method 3, rho = log(2b), but the regularizer is |b|
        b = torch.exp(rho) / 2
        nll_laplacian = diff / b + beta * torch.abs(b)
    elif computing_mode == 4:
        # Method 3, rho = log(2b), but the regularizer is |1/b - 1|
        b = torch.exp(rho) / 2
        nll_laplacian = diff / b + beta * torch.abs(1/b - 1)


    loss = torch.sum(nll_laplacian * mask)
    num_valid_kps = torch.sum(mask).item()
    if num_valid_kps != 0:
        loss /= num_valid_kps

    return loss

def masked_nll_laplacian2(input, target, rho, patches_rho, mask, beta=1.0, gamma=1.0):
    '''
    masked negative log likelihood Laplacian
    :param input: B x N x 2
    :param target: B x N x 2
    :param rho: B x N x 2
    :param patches_logvar: 1 x N x 1
    :param mask: B x N x 1 (or B x N x 2)
    :return:
    '''
    # Method 1, logvar = log(2b), patches_logvar=log(2w)
    twob = torch.exp(rho)
    twow = torch.exp(patches_rho)
    diff = torch.abs(input - target)


    nll_laplacian = 2 * diff / (twob + twow + 1e-6) + beta*torch.log(twob + gamma*twow + 1e-6)

    loss = torch.sum(nll_laplacian * mask)
    num_valid_kps = torch.sum(mask).item()
    if num_valid_kps != 0:
        loss /= num_valid_kps

    return loss


def instance_weighted_nllloss(input, target, instance_weight=None, ignore_index=-1, avg_method=0):
    '''
    Please attention that instance weight is different to class weight.
    This version of nllloss is Faster than version V0
    :param input: N x C
    :param target: N
    :param instance_weight: N
    :param ignore_index: default is -1
    :return: loss
    '''
    ind = (target != ignore_index)  # boolean index

    if avg_method == 0:
        total_num = torch.sum(ind).item()  # method 1
    else:
        total_num = torch.sum(instance_weight[ind]).item()  # method 2

    if instance_weight is not None:
        loss = -torch.sum(input[ind, target[ind]] * instance_weight[ind])
    else:
        loss = -torch.sum(input[ind, target[ind]])
    if total_num > 0:
        loss /= total_num
    return loss

def instance_weighted_nlllossV0(input, target, instance_weight=None, ignore_index=-1, avg_method=0):
    '''
    Please pay attention that instance weight is different to class weight
    :param input: N x C
    :param target: N
    :param instance_weight: N
    :param ignore_index: default is -1
    :return: loss
    '''
    N = input.shape[0]
    container = torch.zeros(N, requires_grad=True).cuda()
    if instance_weight is None:
        instance_weight = torch.ones(N, requires_grad=True).cuda()
    for i in range(N):
        if target[i] == ignore_index:
            instance_weight[i] = 0
            continue
        container[i] = input[i, target[i]]

    container *= instance_weight

    if avg_method == 0:
        # method 1
        total_num = torch.sum(target != ignore_index).item()
    else:
        # method 2
        total_num = torch.sum(instance_weight)

    if total_num == 0:
        return -torch.sum(container)  # which is equal to 0
    return -torch.sum(container) / total_num




