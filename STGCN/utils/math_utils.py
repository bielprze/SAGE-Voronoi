import numpy as np
import tensorflow as tf

def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v) / (v + 1e-5))*100

def MRE(v, v_):
    #return np.mean(np.abs(v_ - v) / (v + 1e-5))
    nonzero = 0
    mre = 0
    for a in range(v.shape[0]):
        for b in range(v.shape[1]):
            if v[a,b,0] > 0.000001:
                mre += (np.abs(v_[a,b,0] - v[a,b,0]) / v[a,b,0])
                nonzero += 1

    if nonzero > 0:
        mre /= nonzero

    mre_list = []
    #return np.sqrt(np.mean((v_ - v) ** 2))

    for a in range(v.shape[0]):
        mre_ = 0
        nonzero = 0
        for b in range(v.shape[1]):
            if v[a,b,0] > 0.000001:
                mre_ += (np.abs(v_[a,b,0] - v[a,b,0]) / v[a,b,0])
                nonzero += 1
        if nonzero > 0:
            mre_list.append(mre_/nonzero)
    mean_mre = sum(mre_list) / len(mre_list)
    return mean_mre

def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    rmse_list = []
    #return np.sqrt(np.mean((v_ - v) ** 2))

    for a in range(v.shape[0]):
        mse = 0
        div = 0
        for b in range(v.shape[1]):
            mse += (v[a, b, 0] - v_[a, b, 0]) * (v[a, b, 0] - v_[a, b, 0])
            div = div + 1
        rmse_list.append(np.sqrt(mse / div))
    mean_rmse = sum(rmse_list) / len(rmse_list)
    xxx = np.sqrt(np.mean((v_ - v) ** 2))
    return mean_rmse

def MSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    mse_list = []
    #return np.sqrt(np.mean((v_ - v) ** 2))

    for a in range(v.shape[0]):
        mse = 0
        div = 0
        for b in range(v.shape[1]):
            mse += (v[a, b, 0] - v_[a, b, 0]) * (v[a, b, 0] - v_[a, b, 0])
            div = div + 1
        mse_list.append(mse / div)
    mean_mse = sum(mse_list) / len(mse_list)
    xxx = np.mean((v_ - v) ** 2)
    return mean_mse

def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    mae_list = []
    #return np.sqrt(np.mean((v_ - v) ** 2))

    for a in range(v.shape[0]):
        mae = 0
        div = 0
        for b in range(v.shape[1]):
            mae += np.abs(v[a, b, 0] - v_[a, b, 0])
            div = div + 1
        mae_list.append(mae / div)
    mean_mse = sum(mae_list) / len(mae_list)
    xxx = np.mean(np.abs(v_ - v))
    return mean_mse

def save_res_to_file(v, v_):
    np.save("results/ref", v[:,:,0])
    np.save("results/stgcn", v_[:, :, 0])

def evaluation(y, y_, x_stats):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)

    if dim == 3:
        # single_step case
        v = z_inverse(y, x_stats['mean'], x_stats['std'])
        v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        save_res_to_file(v, v_)
        #tu są dane!!!
        #return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
        #forecast_horizon, my_seed, mean_mse, sd_mse, mean_rmse, sd_rmse, mean_mae, sd_mae, mean_mre, sd_mre
        return np.array([MSE(v, v_), RMSE(v, v_), MAE(v, v_), MRE(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        y = np.swapaxes(y, 0, 1)
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)

def custom_loss(y_true, y_pred) -> tf.Tensor:
    '''
    Cutom Loss function.
    :param y_true: tf.Tensor, ground truth.
    :param y_pred: tf.Tensor, prediction.
    :return: tf.Tensor, custom loss value (Here L2 loss).
    '''
    return tf.nn.l2_loss(y_true - y_pred)