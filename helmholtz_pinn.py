import sys
#import jax.config as config
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random, jit, vjp, grad, vmap
import jax.flatten_util as flat_utl
from jax.experimental.host_callback import call
from tensorflow_probability.substrates import jax as tfp
from pyDOE import lhs
import time
import functools
from scipy.io import savemat
from pathlib import Path
import scipy.special as sp
from scipy.fft import fft2, fftfreq
import numpy as np

# find the root directory
rootdir = Path(__file__).parent
# change JAX to double precision
jax.config.update('jax_enable_x64', True)

#-------------------------------------------------------------------------------#
# Geometry definition
#-------------------------------------------------------------------------------#
# Geometrical characteristics

L = 2.0   # Length of rectangle (R)
l = 2.0   # Width of rectangle (R)

# We choose l=L to have a square

# Bounds of x, y

x_lower = -L/2.0
x_upper =  L/2.0
y_lower = -l/2.0
y_upper =  l/2.0

R = 0.25  # Radius of circle (S)

# Center of the circle

Cx = 0.0
Cy = 0.0

eps1_r = 1.0    # Real part of electric permittivity outside the disk
eps1_i = 0.0    # Imaginary part of electric permittivity outside the disk

eps2_r = 1.0     # Real part of electric permittivity inside the disk
eps2_i = 0.0     # Imaginary part of electric permittivity inside the disk

mu_r = 1.0

Z_0 = 1.0    # In vacuum (the incident plane wave is injected in vacuum)

v_0_1 = 0.3  # Velocity 1 (outside the disk)
v_0_2 = v_0_1/jnp.sqrt(mu_r*eps2_r) # Velocity 2 (inside the disk) if you change eps2_r, you must also change v_0_2 (in case eps2_r = 4.0, we have v_0_2 = 0.15)

freq  = 0.1  # 0.9 GHz = 900 Mhz

lam1  = v_0_1/freq # Wave length 1 (outside the disk)
lam2  = v_0_2/freq # Wave length 2 (inside the disk)

omk   = (2.0*jnp.pi*freq)/v_0_1 # Pulsation
omk2  = (2.0*jnp.pi*freq)/v_0_2 # Pulsation

kap   = 1.0/(omk*mu_r) # Constant used in the definition of the PDE

ampE  = 1.0 # Amplitude of the electric field
ampH  = ampE/omk # Amplitude of the magnetic field

# We assume that the wave vector has only one component along the x-axis

kx1 = omk   # outside the disk
kx2 = omk2  # inside the disk


def compute_spectrum(residual, grid_size=(256, 256)):
    """
    计算残差的频谱特征
    :param residual: 残差值 (N, 2) 或 (N,)
    :param grid_size: 目标网格大小 (nx, ny)
    :return: (amplitudes, wavevectors, phases)
    """
    # 将残差插值到均匀网格
    x = np.linspace(-1, 1, grid_size[0])
    y = np.linspace(-1, 1, grid_size[1])
    X, Y = np.meshgrid(x, y)
    
    # 创建网格上的残差值
    if residual.shape[1] == 2:  # 实部和虚部
        res_real = residual[:, 0].reshape(grid_size)
        res_imag = residual[:, 1].reshape(grid_size)
        res_grid = res_real + 1j * res_imag
    else:  # 单一残差
        res_grid = residual.reshape(grid_size)
    
    # 计算二维傅里叶变换
    res_fft = fft2(res_grid)
    
    # 计算幅度和相位
    amplitudes = np.abs(res_fft)
    phases = np.angle(res_fft)
    
    # 创建波矢网格
    kx = fftfreq(grid_size[0])
    ky = fftfreq(grid_size[1])
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    wavevectors = np.stack((Kx, Ky), axis=-1)
    
    # 展平并排序
    amplitudes_flat = amplitudes.reshape(-1)
    wavevectors_flat = wavevectors.reshape(-1, 2)
    phases_flat = phases.reshape(-1)
    
    # 按幅度降序排序
    sorted_indices = np.argsort(amplitudes_flat)[::-1]
    
    return (amplitudes_flat[sorted_indices], 
            wavevectors_flat[sorted_indices], 
            phases_flat[sorted_indices])


# initialize the neural network weights and biases
def init_MLP(parent_key, layer_widths, spectrum_info=None, n_f=None):
    """
    :param parent_key: JAX随机密钥
    :param layer_widths: 网络层宽度列表
    :param spectrum_info: 频谱信息 (amplitudes, wavevectors, phases)
    :param n_f: 使用的频率模式数量
    """
    params = []
    keys = random.split(parent_key, num=len(layer_widths) - 1)
    
    # 处理第一层特殊初始化
    if spectrum_info is not None and n_f is not None:
        amplitudes, wavevectors, phases = spectrum_info
        
        # 选择前n_f个主要模式
        n_f = min(n_f, len(amplitudes))
        B_init = wavevectors[:n_f] * 2 * np.pi  # 转换为角频率
        b_init = phases[:n_f]
        
        # 创建第一层参数
        weight_key, bias_key = random.split(keys[0])
        params.append(
            [B_init.T,  # 输入到隐藏层的权重 (2, n_f)
             b_init]    # 偏置 (n_f,)
        )
        # 使用剩余密钥初始化其他层
        keys = keys[1:]
        
        # 调整后续层的输入维度为n_f
        adjusted_widths = [n_f] + layer_widths[2:]
    else:
        # 标准Xavier初始化
        in_dim = layer_widths[0]
        out_dim = layer_widths[1]
        weight_key, bias_key = random.split(keys[0])
        xavier_stddev = jnp.sqrt(2 / (in_dim + out_dim))
        params.append(
            [random.truncated_normal(weight_key, -2, 2, shape=(in_dim, out_dim)) * xavier_stddev,
             random.truncated_normal(bias_key, -2, 2, shape=(out_dim,)) * xavier_stddev]
        )
        keys = keys[1:]
        adjusted_widths = layer_widths[1:]
    
    # 初始化其余层
    for in_dim, out_dim, key in zip(adjusted_widths[:-1], adjusted_widths[1:], keys):
        weight_key, bias_key = random.split(key)
        xavier_stddev = jnp.sqrt(2 / (in_dim + out_dim))
        params.append(
            [random.truncated_normal(weight_key, -2, 2, shape=(in_dim, out_dim)) * xavier_stddev,
             random.truncated_normal(bias_key, -2, 2, shape=(out_dim,)) * xavier_stddev]
        )
    
    return params

# define the basic formation of neural network
def neural_net(params, z, limit, scl, act_s=0, use_spectral_features=False):
    """
    修改后的神经网络前向传播，支持频谱特征
    """
    lb, rb, db, ub = limit[0], limit[1], limit[2], limit[3]
    
    # 归一化输入
    H = jnp.zeros_like(z)
    H = H.at[:, 0].set(2.0 * (z[:, 0] - lb) / (rb - lb) - 1.0)
    H = H.at[:, 1].set(2.0 * (z[:, 1] - db) / (ub - db) - 1.0)
    
    first, *hidden, last = params
    
    # 频谱特征映射
    if use_spectral_features:
        # 应用频谱特征变换: γ(x) = cos(2πBx + b)
        H = jnp.cos(2 * jnp.pi * jnp.dot(H, first[0]) + first[1])
    else:
        # 标准处理
        actv = [jnp.tanh, jnp.sin][act_s]
        H = actv(jnp.dot(H, first[0]) * scl + first[1])
    
    # 中间层处理
    for layer in hidden:
        H = jnp.tanh(jnp.dot(H, layer[0]) + layer[1])
    
    # 输出层
    output = jnp.dot(H, last[0]) + last[1]
    return output


# 计算第一阶段残差
def compute_residual(f_u, grid_size=(256, 256)):
    """计算当前模型的残差"""
    # 创建评估网格
    x = np.linspace(-1, 1, grid_size[0])
    y = np.linspace(-1, 1, grid_size[1])
    X, Y = np.meshgrid(x, y)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    
    # 计算预测值和残差
    prediction = f_u(grid_points)
    analytical = analytical_solution(grid_points)  # 假设有analytical_solution函数
    residual = analytical - prediction
    
    return residual


# generate weights and biases for all variables of CLM problem
def sol_init_MLP(parent_key, n_hl, n_unit):
    '''
    :param n_hl: number of hidden layers [int]
    :param n_unit: number of units in each layer [int]
    '''
    layers = [2] + n_hl * [n_unit] + [2]
    # generate the random key for each network
    keys = random.split(parent_key, 1)
    # generate weights and biases for
    params_u = init_MLP(keys[0], layers)
    return dict(net_u=params_u)


# wrapper to create solution function with given domain size
def sol_pred_create(limit, scl, act_s=0):
    '''
    :param limit: domain size of the input
    :return: function of the solution (a callable)
    '''
    def f_u(params, z):
        # generate the NN
        u = neural_net(params['net_u'], z, limit, scl, act_s)
        return u
    return f_u


def progressive_mNN_create(f_u, limit, scl, alpha, act_s=0):
    """
    渐进式权重融合的神经网络预测函数
    """
    def f_comb(params, z):
        # 当前阶段的预测
        u_now = neural_net(params['net_u'], z, limit, scl, act_s)
        # 前一阶段的预测
        u_prev = f_u(z)
        # 计算动态权重
        weight = 1.0 / (1.0 + alpha * jnp.exp(-jnp.mean(jnp.abs(u_now))))
        # 融合两个阶段的结果
        u = weight * u_now + (1 - weight) * u_prev
        return u
    return f_comb


"""Low-level functions developed for PINN training using JAX"""

# define the mean squared error
def ms_error(diff):
    return jnp.mean(jnp.square(diff), axis=0)


# generate matrix required for vjp for vector gradient
def vgmat(z, n_out, idx=None):
    '''
    :param n_out: number of output variables
    :param idx: indice (list) of the output variable to take the gradient
    '''
    if idx is None:
        idx = range(n_out)
    # obtain the number of index
    n_idx = len(idx)
    # obtain the number of input points
    n_pt = z.shape[0]
    # determine the shape of the gradient matrix
    mat_shape = [n_idx, n_pt, n_out]
    # create the zero matrix based on the shape
    mat = jnp.zeros(mat_shape)
    # choose the associated element in the matrix to 1
    for l, ii in zip(range(n_idx), idx):
        mat = mat.at[l, :, ii].set(1.)
    return mat


# vector gradient of the output with with input
def vectgrad(func, z):
    # obtain the output and the gradient function
    sol, vjp_fn = vjp(func, z)
    # determine the mat grad
    mat = vgmat(z, sol.shape[1])
    # calculate the gradient of each output with respect to each input
    grad_sol = vmap(vjp_fn, in_axes=0)(mat)[0]
    # calculate the total partial derivative of output with input
    n_pd = z.shape[1] * sol.shape[1]
    # reshape the derivative of output with input
    grad_all = grad_sol.transpose(1, 0, 2).reshape(z.shape[0], n_pd)
    return grad_all, sol


# governing equation
def gov_eqn(f_u, z):
    # calculate the output and its derivative with original coordinates
    #u_x, u = vectgrad(f_u, z)

    # calculate the residue of the CCF equation
    fu_1 = lambda z: vectgrad(f_u, z)[0]

    #fu_y = lambda z: vectgrad(f_u, z)[0]
    u_2 = vectgrad(fu_1, z)[0]#此处会产生八列数据，0~3为电场实部对应梯度结果xx xy yx yy，4~7为电场虚部对应梯度结果xx xy yx yy
    d2Ez_r_x2 = u_2[:, 0][:, None]  # d2Ez_r/dx2
    d2Ez_r_y2 = u_2[:, 3][:, None]  # d2Ez_r/dy2

    d2Ez_i_x2 = u_2[:, 4][:, None]  # d2Ez_i/dx2
    d2Ez_i_y2 = u_2[:, 7][:, None]  # d2Ez_i/dy2

    Ez = f_u(z)
    Ez_r = Ez[:, 0:1]
    Ez_i = Ez[:, 1:2]

    x = z[:, 0:1]
    y = z[:, 1:2]

    curl2E_z_r = - (d2Ez_r_x2 + d2Ez_r_y2)
    curl2E_z_i = - (d2Ez_i_x2 + d2Ez_i_y2)

    d2 = (x - Cx) * (x - Cx) + (y - Cy) * (y - Cy)
    d = jnp.sqrt(d2)  # Distance between a point X(x,y) and the center of the disk (Cx,Cy)
    cond = jnp.less(d, R)

    fEz_1 = omk * (eps1_r * Ez_r - eps1_i * Ez_i) - kap * curl2E_z_r  # outside the disk
    fEz_2 = omk * (eps2_r * Ez_r - eps2_i * Ez_i) - kap * curl2E_z_r  # inside the disk

    fEz = jnp.where(cond, fEz_2, fEz_1)

    gEz_1 = -omk * (eps1_r * Ez_i + eps1_i * Ez_r) + kap * curl2E_z_i  # outside the disk
    gEz_2 = -omk * (eps2_r * Ez_i + eps2_i * Ez_r) + kap * curl2E_z_i  # inside the disk

    gEz = jnp.where(cond, gEz_2, gEz_1)
    #fEz_and_gEz = jnp.hstack(fEz, gEz)

    return [fEz, gEz]

def gov_eqn_r(f_u, z):
    # calculate the output and its derivative with original coordinates
    #u_x, u = vectgrad(f_u, z)

    # calculate the residue of the CCF equation
    fu_1 = lambda z: vectgrad(f_u, z)[0]

    #fu_y = lambda z: vectgrad(f_u, z)[0]
    u_2 = vectgrad(fu_1, z)[0]#此处会产生八列数据，0~3为电场实部对应梯度结果xx xy yx yy，4~7为电场虚部对应梯度结果xx xy yx yy
    d2Ez_r_x2 = u_2[:, 0][:, None]  # d2Ez_r/dx2
    d2Ez_r_y2 = u_2[:, 3][:, None]  # d2Ez_r/dy2

    #d2Ez_i_x2 = u_2[:, 4][:, None]  # d2Ez_i/dx2
    #d2Ez_i_y2 = u_2[:, 7][:, None]  # d2Ez_i/dy2

    Ez = f_u(z)
    Ez_r = Ez[:, 0:1]
    Ez_i = Ez[:, 1:2]

    x = z[:, 0:1]
    y = z[:, 1:2]

    curl2E_z_r = - (d2Ez_r_x2 + d2Ez_r_y2)
    #curl2E_z_i = - (d2Ez_i_x2 + d2Ez_i_y2)

    d2 = (x - Cx) * (x - Cx) + (y - Cy) * (y - Cy)
    d = jnp.sqrt(d2)  # Distance between a point X(x,y) and the center of the disk (Cx,Cy)
    cond = jnp.less(d, R)

    fEz_1 = omk * (eps1_r * Ez_r - eps1_i * Ez_i) - kap * curl2E_z_r  # outside the disk
    fEz_2 = omk * (eps2_r * Ez_r - eps2_i * Ez_i) - kap * curl2E_z_r  # inside the disk

    fEz = jnp.where(cond, fEz_2, fEz_1)

    # gEz_1 = -omk * (eps1_r * Ez_i + eps1_i * Ez_r) + kap * curl2E_z_i  # outside the disk
    # gEz_2 = -omk * (eps2_r * Ez_i + eps2_i * Ez_r) + kap * curl2E_z_i  # inside the disk
    #
    # gEz = jnp.where(cond, gEz_2, gEz_1)
    #fEz_and_gEz = jnp.hstack(fEz, gEz)

    return fEz

def gov_eqn_i(f_u, z):
    # calculate the output and its derivative with original coordinates
    #u_x, u = vectgrad(f_u, z)

    # calculate the residue of the CCF equation
    fu_1 = lambda z: vectgrad(f_u, z)[0]

    #fu_y = lambda z: vectgrad(f_u, z)[0]
    u_2 = vectgrad(fu_1, z)[0]#此处会产生八列数据，0~3为电场实部对应梯度结果xx xy yx yy，4~7为电场虚部对应梯度结果xx xy yx yy
    #d2Ez_r_x2 = u_2[:, 0][:, None]  # d2Ez_r/dx2
    #d2Ez_r_y2 = u_2[:, 3][:, None]  # d2Ez_r/dy2

    d2Ez_i_x2 = u_2[:, 4][:, None]  # d2Ez_i/dx2
    d2Ez_i_y2 = u_2[:, 7][:, None]  # d2Ez_i/dy2

    Ez = f_u(z)
    Ez_r = Ez[:, 0:1]
    Ez_i = Ez[:, 1:2]

    x = z[:, 0:1]
    y = z[:, 1:2]

    #curl2E_z_r = - (d2Ez_r_x2 + d2Ez_r_y2)
    curl2E_z_i = - (d2Ez_i_x2 + d2Ez_i_y2)

    d2 = (x - Cx) * (x - Cx) + (y - Cy) * (y - Cy)
    d = jnp.sqrt(d2)  # Distance between a point X(x,y) and the center of the disk (Cx,Cy)
    cond = jnp.less(d, R)

    # fEz_1 = omk * (eps1_r * Ez_r - eps1_i * Ez_i) - kap * curl2E_z_r  # outside the disk
    # fEz_2 = omk * (eps2_r * Ez_r - eps2_i * Ez_i) - kap * curl2E_z_r  # inside the disk
    #
    # fEz = jnp.where(cond, fEz_2, fEz_1)

    gEz_1 = -omk * (eps1_r * Ez_i + eps1_i * Ez_r) + kap * curl2E_z_i  # outside the disk
    gEz_2 = -omk * (eps2_r * Ez_i + eps2_i * Ez_r) + kap * curl2E_z_i  # inside the disk

    gEz = jnp.where(cond, gEz_2, gEz_1)
    #fEz_and_gEz = jnp.hstack(fEz, gEz)

    return gEz

def gov_d3_eqn(f_u, z):  #这个函数理论上应该输出四个df2，两个df，记得看结果
    # allocate the value to each variable
    fc_res = lambda z: gov_eqn(f_u, z)
    # calculate the residue of higher derivative of CCF equation
    dfunc = lambda z: vectgrad(fc_res, z)[0]
    # calculate the residue of the first and second derivative of CCF equation
    d2f, df = vectgrad(dfunc, z)
    return df, d2f

def gov_d3_eqn_r(f_u, z):  #这个函数理论上应该输出四个df2，两个df，记得看结果
    # allocate the value to each variable
    fc_res = lambda z: gov_eqn_r(f_u, z)
    # calculate the residue of higher derivative of CCF equation
    dfunc = lambda z: vectgrad(fc_res, z)[0]
    # calculate the residue of the first and second derivative of CCF equation
    d2f, df = vectgrad(dfunc, z)
    return df, d2f

def gov_d3_eqn_i(f_u, z):  #这个函数理论上应该输出四个df2，两个df，记得看结果
    # allocate the value to each variable
    fc_res = lambda z: gov_eqn_i(f_u, z)
    # calculate the residue of higher derivative of CCF equation
    dfunc = lambda z: vectgrad(fc_res, z)[0]
    # calculate the residue of the first and second derivative of CCF equation
    d2f, df = vectgrad(dfunc, z)
    return df, d2f


def loss_create(predf_u, cond, lw, loss_ref):
    '''
    a function factory to create the loss function based on given info
    :param loss_ref: loss value at the initial of the training
    :return: a loss function (callable)
    '''

    # loss function used for the PINN training
    def loss_fun(params, data):
        # create the function for gradient calculation involves input Z only
        f_u = lambda z: predf_u(params, z)
        # load the data of normalization condition
        z_nm = cond['cond_nm'][0]
        #u_nm = cond['cond_nm'][1]

        # load the position and weight of collocation points
        z_col = data['z_col']
        f_u1 = lambda z: vectgrad(f_u, z)
        # calculate the gradient of phi at origin
        dEz_1, u_nm_p = f_u1(z_nm)
        u_erro1 = EHx_abc_r(z_nm, u_nm_p, dEz_1)
        u_erro2 = EHy_abc_r(z_nm, u_nm_p, dEz_1)
        u_erro3 = EHx_abc_i(z_nm, u_nm_p, dEz_1)
        u_erro4 = EHy_abc_r(z_nm, u_nm_p, dEz_1)

        # calculate the residue of equation
        f = gov_eqn(f_u, z_col)
        # calculate the residue of first and second derivative
        #df_r, d2f_r = gov_d3_eqn_r(f_u, z_col)
        #df_i, d2f_i = gov_d3_eqn_i(f_u, z_col)

        # calculate the mean squared root error of normalization cond.
        norm_err1 = ms_error(u_erro1)
        norm_err2 = ms_error(u_erro2)
        norm_err3 = ms_error(u_erro3)
        norm_err4 = ms_error(u_erro4)
        norm_err = norm_err1 + norm_err2 + norm_err3 + norm_err4
        # calculate the error of far-field exponent cond.
        data_err = jnp.hstack([norm_err])

        # calculate the mean squared root error of equation
        eqn_err_f = ms_error(f[0]) + ms_error(f[1])
        # eqn_err_df_r = ms_error(df_r)
        # eqn_err_d2f_r = ms_error(d2f_r)
        # eqn_err_df_i = ms_error(df_i)
        # eqn_err_d2f_i = ms_error(d2f_i)
        #eqn_err = jnp.hstack([eqn_err_f, eqn_err_df])
        eqn_err = eqn_err_f

        # set the weight for each condition and equation
        data_weight = jnp.array([1.])
        eqn_weight = jnp.array([1., lw[1]])

        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err * data_weight)
        #loss_eqn = jnp.sum(eqn_err * eqn_weight)
        loss_eqn = jnp.sum(eqn_err * 1)

        # calculate the total loss
        loss = (loss_data + lw[0] * loss_eqn) / loss_ref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn]),
                                data_err, eqn_err])
        return loss, loss_info
    return loss_fun


# create the Adam minimizer
@functools.partial(jit, static_argnames=("lossf", "opt"))
def adam_minimizer(lossf, params, data, opt, opt_state):
    """Basic gradient update step based on the opt optimizer."""
    grads, loss_info = grad(lossf, has_aux=True)(params, data)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, loss_info, opt_state


def adam_optimizer(lossf, params, dataf, epoch, lr=1e-3):
    # select the Adam as the minimizer
    opt_Adam = optax.adam(learning_rate=lr)
    # obtain the initial state of the params
    opt_state = opt_Adam.init(params)
    # pre-allocate the loss varaible
    loss_all = []
    data = dataf()
    nc = jnp.int32(jnp.round(epoch / 5))
    nc0 = 2500
    # start the training iteration
    for step in range(epoch):
        # minimize the loss function using Adam
        params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt_Adam, opt_state)
        # print the loss for every 100 iteration
        if step % 100 == 0:
            print(f"Step: {step} | Loss: {loss_info[0]:.4e} |"
                  f" Loss_d: {loss_info[1]:.4e} | Loss_e: {loss_info[2]:.4e}", file=sys.stderr)
            data = dataf()

        # saving the loss
        loss_all.append(loss_info[0:])

        if (step+1) % (2*nc0) == 0:
            lossend = np.array(loss_all[-2 * nc0:])[:, 0]
            lc1 = lossend[0: nc0]
            lc2 = lossend[nc0:]
            mm12 = jnp.abs(jnp.mean(lc1) - jnp.mean(lc2))
            stdl2 = jnp.std(lc2)
            # if the average loss improvement within 'nc' iteration is less than local loss fluctuation (std)
            if mm12 / stdl2 < 0.4:
                # reduce the learning rate by half
                lr = lr / 2
                opt_Adam = optax.adam(learning_rate=lr)
            print(f"learning rate for Adam: {lr:.4e} | mean: {mm12:.3e} | std: {stdl2:.3e}", file=sys.stderr)

    # obtain the total loss in the last iterations
    lossend = jnp.array(loss_all[-nc:])[:, 0]
    # find the minimum loss value
    lmin = jnp.min(lossend)
    # optain the last loss value
    llast = lossend[-1]
    # guarantee the loss value in last iteration is smaller than anyone before
    while llast > lmin:
        params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt_Adam, opt_state)
        llast = loss_info[0]
        # saving the loss
        loss_all.append(loss_info[0:])

    return params, loss_all


# A factory to create a function required by tfp.optimizer.lbfgs_minimize.
def lbfgs_function(lossf, init_params, data):
    _, unflat = flat_utl.ravel_pytree(init_params)

    def update(params_1d):
        params = unflat(params_1d)
        return params

    @jit
    def f(params_1d):
        params = update(params_1d)
        grads, loss_info = grad(lossf, has_aux=True)(params, data)
        grads_1d = flat_utl.ravel_pytree(grads)[0]
        loss_value = loss_info[0]

        f.loss.append(loss_info[0:])  # 直接操作列表，无需call

        # 格式说明：%.4e表示科学计数法保留4位小数，对应loss_info的三个值
        jax.debug.print("Loss: {} | Loss_d: {} | Loss_e: {}",
                        loss_info[0], loss_info[1], loss_info[2])
        return loss_value, grads_1d

    f.update = update
    f.loss = []  # 初始化空列表
    return f


# define the function to apply the L-BFGS optimizer
def lbfgs_optimizer(lossf, params, data, epoch):
    func_lbfgs = lbfgs_function(lossf, params, data)
    # convert initial model parameters to a 1D array
    init_params_1d = flat_utl.ravel_pytree(params)[0]
    # calculate the effective number of iteration
    max_nIter = jnp.int32(epoch / 3)
    # train the model with L-BFGS solver
    results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func_lbfgs, initial_position=init_params_1d,
        tolerance=1e-10, max_iterations=max_nIter)
    params = func_lbfgs.update(results.position)
    # history = func_lbfgs.loss
    num_iter = results.num_objective_evaluations
    loss_all = func_lbfgs.loss
    print(f" Total iterations: {num_iter}")
    return params, loss_all


"""Prepare the collocaiton points"""
def data_func_create(N_col, ds):
    # define the function that can re-sampling for each calling
    def dataf():
        # prepare the collocation points
        x_cn = (2 * lhs(1, N_col) - 1)
        y_cn = (2 * lhs(1, N_col) - 1)
        x_grid, y_grid = np.meshgrid(x_cn, y_cn)
        z_cn = np.vstack((np.ravel(x_grid), np.ravel(y_grid))).T
        #z_cn = jnp.hstack((x_cn, y_cn))
        z_col = jnp.sign(z_cn) * jnp.abs(z_cn) ** 1 * ds * 1.01
        # add the collocation at the boundary
        #z_col = jnp.vstack([z_col, jnp.array([-ds, ds])[:, None]])
        return dict(z_col=z_col)
    return dataf


def EHx_abc_r(z, y, dEz_1):
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual

    nx = 0.0
    ny = 0.0

    nx = jnp.where(z[:, 0:1] == x_lower, -1.0, nx)
    nx = jnp.where(z[:, 0:1] == x_upper, 1.0, nx)

    ny = jnp.where(z[:, 1:2] == y_lower, -1.0, ny)
    ny = jnp.where(z[:, 1:2] == y_upper, 1.0, ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

    dEz_i_x = dEz_1[:, 2:3]
    dEz_i_y = dEz_1[:, 3:4]

    nEx = ny * y[:, 0:1]
    nEy = -nx * y[:, 0:1]

    # Compute H_r from E_i

    # dEz_i_x = dde.grad.jacobian(y, x, i=1, j=0)  # Calculate dEz_i/dx
    # dEz_i_y = dde.grad.jacobian(y, x, i=1, j=1)  # Calculate dEz_i/dy




    # A little reminder : kap = 1/(mu_r*omk)

    Hx = -kap * dEz_i_y  # Hrx function of Ezi Equation (36.1)
    Hy = kap * dEz_i_x  # Hry function of Ezi Equation (36.2)

    # Calculate Hr x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)

    nHz = -nx * Hy + ny * Hx

    # Calculate n x (Hr x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)

    nHxn = ny * nHz
    nHyn = -nx * nHz

    # Calculate the left side of the first equation of Equation (33)

    rEHx = nEx - Z_0 * nHxn  # along the x-axis
    rEHy = nEy - Z_0 * nHyn  # along the y-axis

    # Components of the incident plane wave, wave vector has an only component along the x-axis

    Ezinc = ampE * jnp.cos(kx1 * z[:, 0:1])
    Hyinc = -ampH * kx1 * jnp.cos(kx1 * z[:, 0:1])

    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)

    nExinc = ny * Ezinc
    nEyinc = -nx * Ezinc

    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)

    nHzinc = -nx * Hyinc

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)

    nHxninc = ny * nHzinc
    nHyninc = -nx * nHzinc

    # Calculate the right side of the first equation of Equation (33)

    rEHxinc = nExinc - Z_0 * nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0 * nHyninc  # along the y-axis

    return rEHx - rEHxinc


def EHy_abc_r(x, y, dEz_1):
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual

    nx = 0.0
    ny = 0.0

    nx = jnp.where(x[:, 0:1] == x_lower, -1.0, nx)
    nx = jnp.where(x[:, 0:1] == x_upper, 1.0, nx)

    ny = jnp.where(x[:, 1:2] == y_lower, -1.0, ny)
    ny = jnp.where(x[:, 1:2] == y_upper, 1.0, ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

    nEx = ny * y[:, 0:1]
    nEy = -nx * y[:, 0:1]

    # Compute H_r from E_i

    # dEz_i_x = dde.grad.jacobian(y, x, i=1, j=0)  # Calculate dEz_i/dx
    # dEz_i_y = dde.grad.jacobian(y, x, i=1, j=1)  # Calculate dEz_i/dy
    dEz_i_x = dEz_1[:, 2:3]
    dEz_i_y = dEz_1[:, 3:4]


    # A little reminder : kap = 1/(mu_r*omk)

    Hx = -kap * dEz_i_y  # Hrx function of Ezi Equation (36.1)
    Hy = kap * dEz_i_x  # Hry function of Ezi Equation (36.2)

    # Calculate Hr x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)

    nHz = -nx * Hy + ny * Hx

    # Calculate n x (Hr x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)

    nHxn = ny * nHz
    nHyn = -nx * nHz

    # Calculate the left side of the first equation of Equation (33)

    rEHx = nEx - Z_0 * nHxn  # along the x-axis
    rEHy = nEy - Z_0 * nHyn  # along the y-axis

    # Components (real part) of the incident plane wave, wave vector has an only component along the x-axis

    Ezinc = ampE * jnp.cos(kx1 * x[:, 0:1])
    Hyinc = -ampH * kx1 * jnp.cos(kx1 * x[:, 0:1])

    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)

    nExinc = ny * Ezinc
    nEyinc = -nx * Ezinc

    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)

    nHzinc = -nx * Hyinc

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)

    nHxninc = ny * nHzinc
    nHyninc = -nx * nHzinc

    # Calculate the right side of the first equation of Equation (33)

    rEHxinc = nExinc - Z_0 * nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0 * nHyninc  # along the y-axis

    return rEHy - rEHyinc


# -----------------------#
# Imaginary part of E and H
# -----------------------#

def EHx_abc_i(x, y, dEz_1):
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual

    nx = 0.0
    ny = 0.0

    nx = jnp.where(x[:, 0:1] == x_lower, -1.0, nx)
    nx = jnp.where(x[:, 0:1] == x_upper, 1.0, nx)

    ny = jnp.where(x[:, 1:2] == y_lower, -1.0, ny)
    ny = jnp.where(x[:, 1:2] == y_upper, 1.0, ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

    nEx = ny * y[:, 1:2]
    nEy = -nx * y[:, 1:2]

    # Compute H_i from E_r

    dEz_r_x = dEz_1[:, 0:1] # Calculate dEz_r/dx
    dEz_r_y = dEz_1[:, 1:2]  # Calculate dEz_r/dy

    # A little reminder : kap = 1/(mu_r*omk)

    Hx = kap * dEz_r_y  # Hix function of Ezr Equation (36.3)
    Hy = -kap * dEz_r_x  # Hix function of Ezr Equation (36.4)

    # Calculate Hi x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)

    nHz = -nx * Hy + ny * Hx

    # Calculate n x (Hi x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)

    nHxn = ny * nHz
    nHyn = -nx * nHz

    # Calculate the left side of the second equation of Equation (33)

    rEHx = nEx - Z_0 * nHxn  # along the x-axis
    rEHy = nEy - Z_0 * nHyn  # along the y-axis

    # Components (imaginary part) of the incident plane wave, wave vector has an only component along the x-axis

    Ezinc = ampE * jnp.sin(-kx1 * x[:, 0:1])
    Hyinc = -ampH * kx1 * jnp.sin(-kx1 * x[:, 0:1])

    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)

    nExinc = ny * Ezinc
    nEyinc = -nx * Ezinc

    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)

    nHzinc = -nx * Hyinc

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)

    nHxninc = ny * nHzinc
    nHyninc = -nx * nHzinc

    # Calculate the right side of the second equation of Equation (33)

    rEHxinc = nExinc - Z_0 * nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0 * nHyninc  # along the y-axis

    return rEHx - rEHxinc


def EHy_abc_i(x, y, dEz_1):
    # Calculate normal outgoing vector n=(nx,ny,0) depending on whatever the point (x,y) belongs to the boundary or not
    # x[:,0:1] refers to x coordinate and x[:,1;2] refers to y coordinate like in the defition of the PDE residual

    nx = 0.0
    ny = 0.0

    nx = jnp.where(x[:, 0:1] == x_lower, -1.0, nx)
    nx = jnp.where(x[:, 0:1] == x_upper, 1.0, nx)

    ny = jnp.where(x[:, 1:2] == y_lower, -1.0, ny)
    ny = jnp.where(x[:, 1:2] == y_upper, 1.0, ny)

    # Calculate n x Er : nEx is the component along the x-axis, nEy is the component along the y-axis, nEz=0.0 (Annexe A.1)
    # y[:,0:1] refers to Erz and y[:,1:2] refers to Eiz

    nEx = ny * y[:, 1:2]
    nEy = -nx * y[:, 1:2]

    # Compute H_i from E_r

    dEz_r_x = dEz_1[:, 0:1] # Calculate dEz_r/dx
    dEz_r_y = dEz_1[:, 1:2]  # Calculate dEz_r/dy

    # A little reminder : kap = 1/(mu_r*omk)

    Hx = kap * dEz_r_y  # Hix function of Ezr Equation (36.3)
    Hy = -kap * dEz_r_x  # Hix function of Ezr Equation (36.4)

    # Calculate Hi x n : nHz is the component along the z-axis, nHx=nHy=0.0 (Annexe A.1)

    nHz = -nx * Hy + ny * Hx

    # Calculate n x (Hi x n) : nHxn is the component along the x-axis, nHyn is the component along the y-axis, nHzn=0.0 (Annexe A.1)

    nHxn = ny * nHz
    nHyn = -nx * nHz

    # Calculate the left side of the second equation of Equation (33)

    rEHx = nEx - Z_0 * nHxn  # along the x-axis
    rEHy = nEy - Z_0 * nHyn  # along the y-axis

    # Components (imaginary part) of the incident plane wave, wave vector has an only component along the x-axis

    Ezinc = ampE * jnp.sin(-kx1 * x[:, 0:1])
    Hyinc = -ampH * kx1 * jnp.sin(-kx1 * x[:, 0:1])

    # Calculate n x Erinc : nExinc is the component along the x-axis, nEyinc is the component along the y-axis, nEzinc=0.0 (Annexe A.1)

    nExinc = ny * Ezinc
    nEyinc = -nx * Ezinc

    # Calculate Hrinc x n : nHzinc is the component along the z-axis, nHxinc=nHyinc=0.0 (Annexe A.1)

    nHzinc = -nx * Hyinc

    # Calculate n x (Hrinc x n) : nHxninc is the component along the x-axis, nHyninc is the component along the y-axis, nHzninc=0.0 (Annexe A.1)

    nHxninc = ny * nHzinc
    nHyninc = -nx * nHzinc

    # Calculate the right side of the second equation of Equation (33)

    rEHxinc = nExinc - Z_0 * nHxninc  # along the x-axis
    rEHyinc = nEyinc - Z_0 * nHyninc  # along the y-axis

    return rEHy - rEHyinc

#-------------------------------------------------------------------------------#
# 生成对应的四个边界上的点，用来输入给边界计算函数
#-------------------------------------------------------------------------------#
def up_boundary(n=1000):
    # 边界 u(x,1)=0
    x = (2 * lhs(1, n) - 1)
    y = jnp.ones_like(x)
    #cond = x ** 2 / torch.e
    z = jnp.hstack((x, y))

    return z

def down_boundary(n=1000):
    # 边界 u(x,-1)=0
    x = (2 * lhs(1, n) - 1)
    y = - jnp.ones_like(x)
    z = jnp.hstack((x, y))

    return z

def left_boundary(n=1000):
    # 边界 u(-1,y)=0
    y = (2 * lhs(1, n) - 1)
    x = - jnp.ones_like(y)
    z = jnp.hstack((x, y))

    return z

def right_boundary(n=1000):
    # 边界 u(1,y)=0
    y = (2 * lhs(1, n) - 1)
    x = jnp.ones_like(y)
    z = jnp.hstack((x, y))

    return z

def all_boundary(n):  #获得四个边界上的所有点
    z_nm_l = left_boundary(n)
    z_nm_r = right_boundary(n)
    z_nm_u = up_boundary(n)
    z_nm_d = down_boundary(n)
    z_nm_all = jnp.vstack((z_nm_l, z_nm_r, z_nm_u, z_nm_d))
    return z_nm_all

# Definition of Bessel and Hankel functions

def bessel_function(x, order):
    return sp.jv(order, x)

def bessel_derivative(x, order):
    return sp.jvp(order,x,n=1)

def hankel_first_kind(x, order):
    return sp.hankel1(order, x)

def hankel_first_kind_derivative(x, order):
    return sp.h1vp(order,x,n=1)

def u(r, theta):
    u_i_values = u_i(r, theta)
    u_e_values = u_e(r, theta)

    return np.where(r <= R, u_i_values, u_e_values)


def u_e(r, theta):
    N = 100
    u_e = complex(0.0, 0.0)

    for n in range(1, N + 1):
        i = complex(0, 1)
        m = float(n)

        An1 = mu_r * kx2 * bessel_derivative(-kx2 * R, m) * bessel_function(-kx1 * R, m) - kx1 * bessel_function(
            -kx2 * R, m) * bessel_derivative(-kx1 * R, m)
        An2 = kx1 * hankel_first_kind_derivative(-kx1 * R, m) * bessel_function(-kx2 * R,
                                                                                m) - mu_r * kx2 * bessel_derivative(
            -kx2 * R, m) * hankel_first_kind(-kx1 * R, m)
        An = An1 / An2

        # print("An:", An)

        u_e = u_e + (i ** m) * (bessel_function(-kx1 * r, m) + An * hankel_first_kind(-kx1 * r, m)) * np.cos(m * theta)

        # print("u_e:", u_e )

    A01 = mu_r * kx2 * bessel_derivative(-kx2 * R, 0) * bessel_function(-kx1 * R, 0) - kx1 * bessel_function(-kx2 * R,
                                                                                                             0) * bessel_derivative(
        -kx1 * R, 0)
    A02 = kx1 * hankel_first_kind_derivative(-kx1 * R, 0) * bessel_function(-kx2 * R,
                                                                            0) - mu_r * kx2 * bessel_derivative(
        -kx2 * R, 0) * hankel_first_kind(-kx1 * R, 0)
    A0 = A01 / A02

    u_e = bessel_function(-kx1 * r, 0) + A0 * hankel_first_kind(-kx1 * r, 0) + 2 * u_e

    return u_e


def u_i(r, theta):
    N = 100
    u_i = complex(0.0, 0.0)

    for n in range(1, N + 1):
        i = complex(0, 1)
        m = float(n)

        Bn1 = kx1 * hankel_first_kind_derivative(-kx1 * R, m) * bessel_function(-kx1 * R, m) - kx1 * hankel_first_kind(
            -kx1 * R, m) * bessel_derivative(-kx1 * R, m)
        Bn2 = kx1 * hankel_first_kind_derivative(-kx1 * R, m) * bessel_function(-kx2 * R,
                                                                                m) - mu_r * kx2 * bessel_derivative(
            -kx2 * R, m) * hankel_first_kind(-kx1 * R, m)
        Bn = Bn1 / Bn2

        # print("Bn:", Bn)

        u_i = u_i + (i ** m) * Bn * bessel_function(-kx2 * r, m) * np.cos(m * theta)

        # print("u_i:", u_i)

    B01 = kx1 * hankel_first_kind_derivative(-kx1 * R, 0) * bessel_function(-kx1 * R, 0) - kx1 * hankel_first_kind(
        -kx1 * R, 0) * bessel_derivative(-kx1 * R, 0)
    B02 = kx1 * hankel_first_kind_derivative(-kx1 * R, 0) * bessel_function(-kx2 * R,
                                                                            0) - mu_r * kx2 * bessel_derivative(
        -kx2 * R, 0) * hankel_first_kind(-kx1 * R, 0)
    B0 = B01 / B02

    u_i = B0 * bessel_function(-kx2 * r, 0) + 2 * u_i

    return u_i

"""Set the conditions of the problem"""

# select the random seed
seed = 1234
key = random.PRNGKey(seed)
np.random.seed(seed)

# create the subkeys
keys = random.split(key, 4)

# select the size of neural network
n_hl = 4
n_unit = 20
scl = 1

# number of sampling points
N_col = 31

# set the size of domain
ds = 1.
lmt = jnp.array([[-ds], [ds], [-ds], [ds]])


# set the training iteration
epoch1 = 10000
epoch2 = 10000
lw = [0.01, 0.2]

# initialize the weights and biases of the network
trained_params = sol_init_MLP(keys[0], n_hl, n_unit)

# prepare the normalization condition
#z_nm = jnp.array([[0.]])
boundary_num = 21
# z_nm_l, cond_nm_l = left_boundary(boundary_num)
# z_nm_r, cond_nm_r = right_boundary(boundary_num)
# z_nm_u, cond_nm_u = up_boundary(boundary_num)
# z_nm_d, cond_nm_d = down_boundary(boundary_num)
# z_nm = jnp.vstack((z_nm_l, z_nm_r, z_nm_u, z_nm_d))
# cond_nm = jnp.vstack((cond_nm_l, cond_nm_r, cond_nm_u, cond_nm_d))
#cond_nm = jnp.array([[0.]])
z_nm = all_boundary(boundary_num)
cond = dict(cond_nm=[z_nm])

# prepare the collocation points to evaluate equation gradient
dataf = data_func_create(N_col, ds)
# group all the conditions and collocation points
data = dataf()

# create the solution function
pred_u = sol_pred_create(lmt, scl, act_s=1)  #此处由0被修改成1，代表第一隐藏层的激活函数改为sin函数

# calculate the loss function
NN_loss = loss_create(pred_u, cond, lw, loss_ref=1)
loss0 = NN_loss(trained_params, data)[0]

"""First stage of training"""

# set the learning rate for Adam
lr = 1e-3
# training the neural network
start_time = time.time()
trained_params, loss1 = adam_optimizer(NN_loss, trained_params, dataf, epoch1, lr=lr)
data = dataf()
trained_params, loss2 = lbfgs_optimizer(NN_loss, trained_params, data, epoch2)


# calculate the equation residue
f_u1 = lambda z: pred_u(trained_params, z)
fu1_x = lambda z: vectgrad(f_u1, z)[0]

# calculate the solution
nbx = 101
nby = 101

# Stack x, y coordinates
x_star0 = np.linspace(-1, 1, nbx)
y_star0 = np.linspace(-1, 1, nby)
x_star = jnp.sign(x_star0) * jnp.abs(x_star0) ** 1 * 1
y_star = jnp.sign(y_star0) * jnp.abs(y_star0) ** 1 * 1

x_grid, y_grid = np.meshgrid(x_star, y_star)
z_star = np.vstack((np.ravel(x_grid), np.ravel(y_grid))).T

#z_star = jnp.hstack((x_star, y_star))

u1_p = f_u1(z_star)
#u1_xx, u1_x = vectgrad(fu1_x, z_star)

f1_p = gov_eqn(f_u1, z_star)
#df1, d2f1 = gov_d3_eqn(f_u1, z_star)

# generate the last loss
loss_all = (loss1+loss2)
#loss_all = loss1

#%%
"""second stage of training - 使用频谱感知初始化"""

# 计算第一阶段残差
print("Computing residual for stage 1...")
# 创建评估网格
x_eval = np.linspace(-1, 1, 256)
y_eval = np.linspace(-1, 1, 256)
X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
z_eval = np.vstack((X_eval.ravel(), Y_eval.ravel())).T

# 计算解析解
cartesian_u_values = u(np.sqrt(X_eval**2 + Y_eval**2), np.arctan2(Y_eval, X_eval))
cartesian_u_values = np.nan_to_num(cartesian_u_values)
Erz_real = np.real(cartesian_u_values).reshape(-1, 1)
Eiz_real = np.imag(cartesian_u_values).reshape(-1, 1)
analytical_solution = np.hstack((Erz_real, Eiz_real))

# 计算预测值
u1_pred = f_u1(z_eval)

# 计算残差
residual1 = analytical_solution - u1_pred

# 计算频谱特征
amplitudes, wavevectors, phases = compute_spectrum(residual1)
spectrum_info = (amplitudes, wavevectors, phases)
n_f = min(1000, len(amplitudes))  # 使用前1000个主要模式


# idxZero = np.where(f1_p[0:-1, 0] * f1_p[1:, 0] < 0)[0]
# NumZero = idxZero.shape[0]
scl2 = np.pi  #3*NumZero + 1
alpha2 = 2.0  # 权重调节参数
epsil = jnp.array([0.05])

# set the training iteration
epoch1 = 10000
epoch2 = 10000
lw = [0.002, 0.001]

# prepare the collocation points to evaluate equation gradient
dataf2 = data_func_create(N_col*2, ds)
# group all the conditions and collocation points
data2 = dataf2()

# 使用频谱信息初始化第二阶段网络
print(f"Initializing stage 2 with {n_f} spectral components...")

# initialize the weights and biases of the network
#trained_params2 = sol_init_MLP(keys[1], n_hl, n_unit)
trained_params2 = {'net_u': init_MLP(keys[1], [2] + n_hl * [n_unit] + [2], 
                          spectrum_info, n_f)}
# create the solution function
#pred_u2 = progressive_mNN_create(f_u1, lmt, scl2, alpha2, act_s=1)
# 修改第二阶段预测函数 - 使用频谱特征
def pred_u2(params, z):
    return f_u1(z) + neural_net(params['net_u'], z, lmt, scl2, act_s=1, use_spectral_features=True)


# calculate the loss function
NN_loss2 = loss_create(pred_u2, cond, lw, loss_ref=1)
loss0 = NN_loss2(trained_params2, data2)[0]
NN_loss2 = loss_create(pred_u2, cond, lw, loss_ref=loss0)

# training the neural network
trained_params2, loss1 = adam_optimizer(NN_loss2, trained_params2, dataf2, epoch1, lr=lr)
data2 = dataf2()
trained_params2, loss2 = lbfgs_optimizer(NN_loss2, trained_params2, data2, epoch2)

# calculate the equation residue
f_u2 = lambda z: pred_u2(trained_params2, z)
fu2_x = lambda z: vectgrad(f_u2, z)[0]

# calculate the solution
u2_p = f_u2(z_star)
#u2_xx, u2_x = vectgrad(fu2_x, z_star)

f2_p = gov_eqn(f_u2, z_star)
#df2, d2f2 = gov_d3_eqn(f_u2, z_star)

# generate the last loss
loss_all = loss_all + (loss1+loss2)
#loss_all = loss_all + loss1

#%%
"""Third stage of training - 使用频谱感知初始化"""

# 计算第二阶段残差
print("Computing residual for stage 2...")
u2_pred = f_u2(z_eval)
residual2 = analytical_solution - u2_pred

# 计算频谱特征
amplitudes2, wavevectors2, phases2 = compute_spectrum(residual2)
spectrum_info2 = (amplitudes2, wavevectors2, phases2)
n_f2 = min(1000, len(amplitudes2))  # 使用前1000个主要模式

# idxZero2 = jnp.where(f2_p[0:-1, 0] * f2_p[1:, 0] < 0)[0]
# NumZero2 = idxZero2.shape[0]
#scl3 = 3*NumZero2 + 1
scl3 = np.pi * np.pi
alpha3 = 1.0  # 权重调节参数
epsil = jnp.array([0.01])

# prepare the collocation points to evaluate equation gradient
dataf3 = data_func_create(N_col*4, ds)
# group all the conditions and collocation points
data3 = dataf3()

# set the training iteration
epoch1 = 20000
lw = [0.0001, 0.000002]

# initialize the weights and biases of the network
#trained_params3 = sol_init_MLP(keys[2], n_hl, n_unit)
# 使用频谱信息初始化第三阶段网络
print(f"Initializing stage 3 with {n_f2} spectral components...")
trained_params3 = {'net_u': init_MLP(keys[2], [2] + n_hl * [n_unit] + [2], 
                          spectrum_info2, n_f2)}

# 修改第三阶段预测函数 - 使用频谱特征
def pred_u3(params, z):
    return f_u2(z) + neural_net(params['net_u'], z, lmt, scl3, act_s=1, use_spectral_features=True)

# 创建第三阶段的预测函数
#pred_u3 = progressive_mNN_create(f_u2, lmt, scl3, alpha3, act_s=1)

# calculate the loss function
NN_loss3 = loss_create(pred_u3, cond, lw, loss_ref=1)
loss0 = NN_loss3(trained_params3, data3)[0]
NN_loss3 = loss_create(pred_u3, cond, lw, loss_ref=loss0)

# training the neural network
trained_params3, loss1 = adam_optimizer(NN_loss3, trained_params3, dataf3, epoch1, lr=lr)

# calculate the equation residue
f_u3 = lambda z: pred_u3(trained_params3, z)
fu3_x = lambda z: vectgrad(f_u3, z)[0]

# calculate the solution
u3_p = f_u3(z_star)
#u3_xx, u3_x = vectgrad(fu3_x, z_star)

f3_p = gov_eqn(f_u3, z_star)
#df3, d2f3 = gov_d3_eqn(f_u3, z_star)

# generate the last loss
loss_all = loss_all + loss1

#%%
import matplotlib.pyplot as plt

def draw_and_save_figure(x_grid, y_grid, u_p_Eiz_fig, Eiz_real_fig, Eiz_error_fig, u_p_Erz_fig, Erz_real_fig, Erz_error_fig, figname):
    fig, ax = plt.subplots(2, 3, figsize=(12, 12))

    # Add a main title to the entire figure
    fig.suptitle(figname, fontsize=16)

    # Draw Eiz_pred

    axp0 = ax[0, 0].pcolor(x_grid, y_grid, u_p_Eiz_fig, cmap='seismic', shading='auto')
    cbar0 = fig.colorbar(axp0, ax=ax[0, 0], shrink=0.5)
    ax[0, 0].set_xlabel('x')
    ax[0, 0].set_ylabel('y')
    ax[0, 0].set_title('Eiz_pred')
    ax[0, 0].set_aspect('equal')

    # Draw Eiz_exact

    axp1 = ax[0, 1].pcolor(x_grid, y_grid, Eiz_real_fig, cmap='seismic', shading='auto')
    cbar0 = fig.colorbar(axp1, ax=ax[0, 1], shrink=0.5)
    ax[0, 1].set_xlabel('x')
    ax[0, 1].set_ylabel('y')
    ax[0, 1].set_title('Eiz_exact')
    ax[0, 1].set_aspect('equal')

    # Draw Eiz_error

    axp2 = ax[0, 2].pcolor(x_grid, y_grid, Eiz_error_fig, cmap='seismic', shading='auto')
    cbar0 = fig.colorbar(axp2, ax=ax[0, 2], shrink=0.5)
    ax[0, 2].set_xlabel('x')
    ax[0, 2].set_ylabel('y')
    ax[0, 2].set_title('Eiz_error')
    ax[0, 2].set_aspect('equal')

    # Draw Erz_pred

    axp3 = ax[1, 0].pcolor(x_grid, y_grid, u_p_Erz_fig, cmap='seismic', shading='auto')
    cbar1 = fig.colorbar(axp3, ax=ax[1, 0], shrink=0.5)
    ax[1, 0].set_xlabel('x')
    ax[1, 0].set_ylabel('y')
    ax[1, 0].set_title('Erz_pred')
    ax[1, 0].set_aspect('equal')

    # Draw Erz_exact

    axp4 = ax[1, 1].pcolor(x_grid, y_grid, Erz_real_fig, cmap='seismic', shading='auto')
    cbar1 = fig.colorbar(axp4, ax=ax[1, 1], shrink=0.5)
    ax[1, 1].set_xlabel('x')
    ax[1, 1].set_ylabel('y')
    ax[1, 1].set_title('Erz_exact')
    ax[1, 1].set_aspect('equal')

    # Draw Erz_error

    axp5 = ax[1, 2].pcolor(x_grid, y_grid, Erz_error_fig, cmap='seismic', shading='auto')
    cbar1 = fig.colorbar(axp5, ax=ax[1, 2], shrink=0.5)
    ax[1, 2].set_xlabel('x')
    ax[1, 2].set_ylabel('y')
    ax[1, 2].set_title('Erz_error')
    ax[1, 2].set_aspect('equal')

    filename = figname + ".png"
    fig.savefig(filename)
    plt.show()

######################################################################
############################# Plotting ###############################
######################################################################
# 获得三个阶段的结果展示数据
u1_p_Erz_fig = u1_p[:, 0].reshape(nbx, nby)
u1_p_Eiz_fig = u1_p[:, 1].reshape(nbx, nby)
u2_p_Erz_fig = u2_p[:, 0].reshape(nbx, nby)
u2_p_Eiz_fig = u2_p[:, 1].reshape(nbx, nby)
u3_p_Erz_fig = u3_p[:, 0].reshape(nbx, nby)
u3_p_Eiz_fig = u3_p[:, 1].reshape(nbx, nby)

# 计算并转换真实值计算结果
# Calculate cartesian_u_values using the function u

cartesian_u_values = u(np.sqrt(x_grid**2 + y_grid**2), np.arctan2(y_grid, x_grid))
cartesian_u_values = np.nan_to_num(cartesian_u_values)

# Extract real and imaginary parts of the values

Erz_real_fig = np.real(cartesian_u_values)
Eiz_real_fig = np.imag(cartesian_u_values)


# 计算并转换误差值

Erz_error1_fig = np.abs(Erz_real_fig-u1_p_Erz_fig)
Eiz_error1_fig = np.abs(Eiz_real_fig-u1_p_Eiz_fig)
Erz_error2_fig = np.abs(Erz_real_fig-u2_p_Erz_fig)
Eiz_error2_fig = np.abs(Eiz_real_fig-u2_p_Eiz_fig)
Erz_error3_fig = np.abs(Erz_real_fig-u3_p_Erz_fig)
Eiz_error3_fig = np.abs(Eiz_real_fig-u3_p_Eiz_fig)

# 第一次绘制并保存
draw_and_save_figure(x_grid, y_grid, u1_p_Eiz_fig, Eiz_real_fig, Eiz_error1_fig, u1_p_Erz_fig, Erz_real_fig, Erz_error1_fig, "First_Stage of Plots")

# 第二次绘制并保存
draw_and_save_figure(x_grid, y_grid, u2_p_Eiz_fig, Eiz_real_fig, Eiz_error2_fig, u2_p_Erz_fig, Erz_real_fig, Erz_error2_fig, "Second Stage of Plots")

# 第三次绘制并保存
draw_and_save_figure(x_grid, y_grid, u3_p_Eiz_fig, Eiz_real_fig, Eiz_error3_fig, u3_p_Erz_fig, Erz_real_fig, Erz_error3_fig, "Third Stage of Plots")

