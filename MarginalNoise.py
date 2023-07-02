#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def like(hyp, ModelInfo):
    x_b = ModelInfo['x_b']
    x_u = ModelInfo['x_u']
    x_v = ModelInfo['x_v']

    u_b = ModelInfo['u_b']
    u = ModelInfo['u']
    v = ModelInfo['v']

    y = np.concatenate([u_b, u, v])

    sigma_u = np.exp(hyp[-2]).astype(object)
    sigma_v = np.exp(hyp[-1]).astype(object)

    jitter = ModelInfo['jitter']

    N_b, D = x_b.shape
    N_u = u.shape[0]
    N_v = v.shape[0]
    N = N_b + N_u + N_v

   
    #print(x_b.shape, x_u.shape, x_v.shape)
    K_nn_uu = k_nn_uu(x_b, x_b, hyp[:-2], 0,ModelInfo) + np.eye(N_b)*jitter
    K_nn1_uu = k_nn1_uu(x_b, x_u, hyp[:-2], 0,ModelInfo)
    K_nn1_uv = k_nn1_uv(x_b, x_v, hyp[:-2], 0,ModelInfo)

    K_n1n1_uu = k_n1n1_uu(x_u, x_u, hyp[:-2], 0,ModelInfo) + np.eye(N_u)*sigma_u + np.eye(N_u)*jitter
    K_n1n1_uv = k_n1n1_uv(x_u, x_v, hyp[:-2], 0,ModelInfo)

    K_n1n1_vv = k_n1n1_vv(x_v, x_v, hyp[:-2], 0,ModelInfo) + np.eye(N_v)*sigma_v + np.eye(N_v)*jitter
    

    K = np.block([[K_nn_uu, K_nn1_uu, K_nn1_uv],
              [K_nn1_uu.T, K_n1n1_uu, K_n1n1_uv],
              [K_nn1_uv.T, K_n1n1_uv.T, K_n1n1_vv]])
    
   
    #print(K.shape)
    K=npd(K)
    
    L = np.linalg.cholesky(K)
    
    #global ModelInfo
    ModelInfo['L']=L
    
    lambda_sq = np.exp(hyp[-2]).astype(object)
    sigma_sq = np.exp(hyp[-1]).astype(object)
    
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    
    # Sort eigenvalues and eigenvectors in descending order
    
    s = eigenvalues[np.argsort(eigenvalues)[::-1]]    
    #s = np.concatenate([[0], eigenvalues])
    
    # Construct the diagonal eigenvalue matrix S
    S = np.diag(eigenvalues)
    
    # Compute U matrix
    U = eigenvectors
    y_tilde = U.T @ y

    d=np.ones(N)
    g=np.ones(N)
    dlgdi_dsig=np.ones(N)
    dlgdi_dlam=np.ones(N)
    dgi_dsig=np.ones(N)
    dgi_dlamb=np.ones(N)
    
    for i in range(N):
     
     d[i] = s[i]/ (s[i] + (sigma_sq/(lambda_sq+1e-8)) ) + 1
     
     g[i] = ((d[i] ** 2) + 4)/(sigma_sq * d[i]) 
     
     dlgdi_dsig[i] = (1 / (sigma_sq + 2*lambda_sq * s[i])) - (1 / (sigma_sq + lambda_sq*s[i]))
    
     dlgdi_dlam[i] = (s[i] * sigma_sq) / ((sigma_sq + lambda_sq * s[i]) * (sigma_sq + 2 * lambda_sq * s[i]))
    
     dgi_dsig[i] = -(4 / sigma_sq*2) - (sigma_sq4 - 2 * lambda_sq2 * s[i]**2 * sigma_sq*2) / (
            sigma_sq**2 * (sigma_sq + lambda_sq * s[i])**2 * (sigma_sq + 2 * lambda_sq * s[i])**2)
    
     dgi_dlamb[i] = (s[i] / (sigma_sq + lambda_sq * s[i])**2) - (4 * s[i] / (sigma_sq + 2 * lambda_sq * s[i])**2)
    
    Ly = N * np.log(sigma_sq) + np.sum(np.log(d) + (y_tilde ** 2)*g) - (4 * np.dot(y.T, y))/sigma_sq
    
    dLy_dsig = N / sigma_sq + (4 * np.dot(y.T, y))/sigma_sq*2 + np.sum(dlgdi_dsig + y_tilde*2 * dgi_dsig)
    
    dLy_dlamb = np.sum(dlgdi_dlam + (y_tilde**2) *dgi_dlamb)
    NLML=(Ly.real)
    D_NLML = np.zeros_like(hyp)

    D_NLML[4] =  (dLy_dlamb.real)
    D_NLML[5] = (dLy_dsig[0][0].real)
     
    return NLML, D_NLML

