import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


def get_connectivity(data, alpha):
    X1 = np.corrcoef(data, rowvar=False)**alpha
    np.fill_diagonal(X1, 0)
    L1 = np.diag(np.sum(X1, axis=1))
    Penalty = L1 - X1
    return Penalty

def s2cca(data, para, u0, v0):
    # 초기 설정
    n_snp = data['X'][0].shape[1] 
    n_img = data['Y'][0].shape[1] 
    n_class = data['class']
    n = data['X'][0].shape[0]
    
    # 데이터 설정
    X = data['X']
    Y = data['Y']

    # 연결성 벌칙
    H1 = [data['PX'][0]]
    H1 = np.array(H1).reshape((n_snp,n_snp))
    H2 = [data['PY'][0], data['PY'][1]]
    H2 = np.array(H2)


    # 초기화
    u_new = u0[:,np.newaxis]
    v0 = v0[:, np.newaxis]
    v = [v0, v0]
    v = np.array(v)

    eps = np.finfo(float).eps
    d1 = 1.0 / np.sqrt(u_new**2 + eps)  # u에 대한 가중치
    d2 = [1.0/(np.sqrt(v[0,:,:]**2) + eps), 1.0/(np.sqrt(v[1,:,:]**2) + eps)] # v에 대한 가중치

    max_iter = 400
    err = 0.005
    diff_obj = err * 10

    # 파라미터 설정
    lambda1 = para[2]
    lambda2 = [para[3], para[4]]
    tau = para[5]
    objFun = [0]*max_iter
    for i in range(max_iter):

        a = 0

        beta_1 = para[0]
        beta_2 = para[1]

        # U 해결, V 고정
        u = u0[:, np.newaxis]

        for ic in range(n_class):
            u = u + (X[ic].T @ ((Y[ic]@v[ic,:,:])/n_img))
        
        u = u + lambda1*np.diag(np.squeeze(H1@u_new))@(np.diag(np.squeeze(d1))@u_new)

        # U 업데이트
        u_new = soft(u, int(beta_1))
        u = u_new
        
        d1 = 1.0 / np.sqrt(u**2 + eps)

        # V 해결, U 고정

        for ic in range(n_class):
            
            v[ic,:,:] = Y[ic].T @ (X[ic] @ u / n_snp) + lambda2[ic] * np.diag(np.squeeze(H2[ic] @ v[ic,:,:]))@(np.diag(np.squeeze(d2[ic]))@v[ic,:,:])

        v_new, _ = flsa_2c(v, tau)
        v_new = vsoft(v_new, int(beta_2))
        v = v_new
        
        d2 = [1.0/(np.sqrt(v[0,:,:]**2) + eps), 1.0/(np.sqrt(v[1,:,:]**2) + eps)]
        
        # Objective Function 계산
        # print(u)
        # print(v[0,:,:])
        # print(v[1,:,:])
        
        for ic in range(n_class):

            objFun[i] = (-u.T@X[ic].T@Y[ic]@v[ic,:,:]) + lambda2[ic] * v[ic,:,:].T @ H2[ic]@v[ic,:,:]+ beta_2 * np.sum(np.abs(v[ic,:,:]))

        objFun[i] = objFun[i]+beta_1*np.sum(np.abs(u)) + tau * np.sum(np.abs(v[0,:,:] - v[1,:,:])) + lambda1 * u.T@H1@u
    

        if i != 0:
            diff_obj = np.abs((objFun[i] - objFun[i-1]) / objFun[i-1]) # relative prediction error
            #plt.plot(i, diff_obj, 'o')
            print('diff_obj: ',diff_obj)
            
        
        if diff_obj < err:
            #hold off
            break

        obj = objFun

    return u, v, obj

# function to solve fused lasso in two-class cases
def flsa_2c(v, lam):
    dx = (v[0, :, :] - v[1, :, :]) / 2
    dx = np.sign(dx) * np.minimum(np.abs(dx), lam)
    vf = np.copy(v)
    vf[0, :, :] = v[0, :, :] - dx
    vf[1, :, :] = v[1, :, :] + dx
    vhat = vf
    return vf, vhat

# soft thresholding with normalization
def soft(x, lam):
    n = x.shape[0]
    abs_x = np.squeeze(np.abs(x))
    temp = np.sort(abs_x)[::-1]
    th = temp[int(lam) - 1]
    y = np.sign(x) * np.maximum(np.abs(x) - np.tile(th,(n,1)), 0)
    ny = np.sqrt(np.sum(y**2))
    y = y / ny
    return y

# v soft thresholding with normalization
def vsoft(x, lam):
    # 2,90,1
    nc, n, k = x.shape
    th_pos = int(lam)
    th_pos = np.repeat(th_pos, k * nc)
    th_pos = th_pos.astype(int)
    x = x.reshape((n, k * nc), order='C')
    abs_x = np.abs(x)
    sort_indices = np.argsort(abs_x, axis=0)[::-1]
    temp = np.take_along_axis(abs_x, sort_indices, axis=0)
    th = temp[th_pos[0]][np.newaxis,:]
    y = np.squeeze(np.sign(x)) * np.maximum((np.abs(x) - np.tile(th,(n, 1))), 0)
    v = y.reshape(nc, n, k)
    fnorm = np.sqrt(np.sum(np.sum(v**2, axis=0), axis=0))
    v = v / np.tile(fnorm, (nc,n,k))
    return v


scaler = StandardScaler()
mmscaler = MinMaxScaler()
u0_gt = np.concatenate((np.ones(3), np.zeros(25), np.ones(2), np.zeros(20), np.ones(10), np.zeros(20),
                        np.ones(2), np.zeros(25), np.ones(3), np.zeros(10), np.ones(10), np.zeros(10),
                        np.ones(10), np.zeros(50), np.ones(50), np.zeros(170),np.ones(3), np.zeros(25), np.ones(2), np.zeros(20), np.ones(10), np.zeros(20),
                        np.ones(2), np.zeros(25), np.ones(3), np.zeros(10), np.ones(10), np.zeros(10),
                        np.ones(10), np.zeros(50), np.ones(50), np.zeros(150))) #820
v0_gt = np.concatenate((np.ones(1), np.zeros(20), np.ones(2), np.zeros(20), np.ones(2), np.zeros(20),
                        np.ones(5), np.zeros(20), np.ones(10),np.zeros(10), np.ones(10), np.zeros(10), np.ones(10), np.zeros(20))) #160
v1_gt = np.concatenate((np.ones(1), np.zeros(35), np.ones(2), np.zeros(5), np.ones(2), np.zeros(20),
                        np.ones(5), np.zeros(20), np.ones(10), np.zeros(30), np.ones(20), np.zeros(10))) #160


        
indices = np.where((v0_gt==v1_gt)&(v0_gt==1))
print(indices)
pairs = list(itertools.combinations(indices[0],2))

n = 160

# 0-80 사이의 랜덤한 값으로 채워진 정방행렬 생성
matrix = np.zeros((n, n))

# 대칭행렬을 만들기 위해 상삼각행렬과 하삼각행렬의 평균을 취함
matrix = (matrix + matrix.T) / 2

# 대각선 요소를 0으로 설정
np.fill_diagonal(matrix, 0)

for p in pairs:
    matrix[p] = 1

matrix = mmscaler.fit_transform(matrix)
con = matrix

a = [0, 43, 44, 45, 46, 65, 66, 67, 68, 69,81,88,89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,150,151,152,153]

b = [0, 43, 44, 47, 48,50,55,57, 65, 66, 67, 68, 69, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,112,113,114,115,116, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139]

v0_test = np.zeros(160, dtype=float)
v1_test = np.zeros(160, dtype=float)

v0_test[a] = 1
v1_test[b] = 1
v0_gt = np.array(v0_test)
v1_gt = np.array(v1_test)
u0_gt = np.concatenate((np.ones(5), np.zeros(25), np.ones(10), np.zeros(20), np.ones(5), np.zeros(20), 
                        np.ones(5), np.zeros(35), np.ones(30), np.zeros(50), np.ones(50), np.zeros(40), 
                        np.ones(40), np.zeros(10), np.ones(10), np.zeros(100),np.ones(10), np.zeros(45), np.ones(20), np.zeros(20), np.ones(5), np.zeros(25),
                        np.ones(20), np.zeros(25), np.ones(30), np.zeros(70), np.ones(20), np.zeros(10),
                        np.ones(10), np.zeros(20), np.zeros(35))) 


scaler = StandardScaler()

# Setting initial information
n = 100 #sample 
p = 820 #SNP 
q = 160 #neuro feature
class_val = 2

sig_noise = 3 # Change if necessary

# Generate latent variable
z = np.random.randn(n)
z = np.sign(z) * (np.abs(z) + 0.1)
z = z.reshape(100,1)
z = scaler.fit_transform(z)
z1 = np.random.randn(n)
z1 = np.sign(z1) * (np.abs(z1) + 0.1)
z1 = z1.reshape(100,1)
z1 = scaler.fit_transform(z1)

# Generate X and Y
data = {}
data['X'] = [np.dot(z, u0_gt.reshape(1, -1)) + np.random.randn(n, p) * sig_noise, np.dot(z, u0_gt.reshape(1, -1)) + np.random.randn(n, p) * sig_noise]
data['Y'] = [np.dot(z, v0_gt.reshape(1, -1)) + np.random.randn(n, q) * sig_noise, np.dot(z, v1_gt.reshape(1, -1)) + np.random.randn(n, q) * sig_noise]
data['class'] = class_val

# Calculate connectivity penalty
data['PX'] = [get_connectivity(data['X'][0], 2), get_connectivity(data['X'][1], 2)]
data['PY'] = [con,con]

# Normalization
data['X'] = [data['X'][0] / np.std(data['X'][0]), data['X'][1] / np.std(data['X'][1])]
data['Y'] = [data['Y'][0] / np.std(data['Y'][0]), data['Y'][1] / np.std(data['Y'][1])]

# Run joint connectivity sparse CCA
para = [230,50,0.1,0.1,0.2,0.1]  # Change if necessary
u0 = np.ones(p) / p # 420
v0 = np.ones(q) / q # 160

u, v, obj = s2cca(data, para, u0, v0)
