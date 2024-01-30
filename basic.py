import scipy.interpolate as interpolate
import numpy as np
import scipy
from require import *

if muscle_number == 3:
    if left_or_right == 'left':
        muscle_idx = ['bic_s_l', 'brachialis_1_l', 'brachiorad_1_l']
    else:
        assert left_or_right == 'right'
        muscle_idx = ['bic_s', 'brachialis_1', 'brachiorad_1']
    OptimalFiberLength = [0.10740956509395153, 0.0805163842230218, 0.15631665675596404]
    MaximumPennationAngle = [1.4706289056333368, 1.4706289056333368, 1.4706289056333368]
    PennationAngleAtOptimal = [0, 0, 0]
    KshapeActive = [0.45, 0.45, 0.45]
else:
    assert muscle_number == 4
    if left_or_right == 'left':
        muscle_idx = ['bic_s_l', 'brachialis_1_l', 'brachiorad_1_l', 'tric_long_1_l']
    else:
        assert left_or_right == 'right'
        muscle_idx = ['bic_s', 'brachialis_1', 'brachiorad_1', 'tric_long_1']
    OptimalFiberLength = [0.10740956509395153, 0.0805163842230218, 0.15631665675596404, 0.09027865826431776]
    MaximumPennationAngle = [1.4706289056333368, 1.4706289056333368, 1.4706289056333368, 1.4706289056333368]
    PennationAngleAtOptimal = [0, 0, 0, 0.174532925199]
    KshapeActive = [0.45, 0.45, 0.45, 0.45]


def resample_by_len(orig_list: list, target_len: int):
    '''
    同于标准重采样，此函数将len(list1)=x 从采样为len(list2)=y；y为指定的值，list2为输出
    :param orig_list: 是list,重采样的源列表：list1
    :param target_len: 重采样的帧数：y
    :return: 重采样后的数组:list2
    '''
    orig_list_len = len(orig_list)
    # print(orig_list_len)
    k = target_len / orig_list_len
    x = [x * k for x in range(orig_list_len)]
    x[-1] = 3572740
    if x[-1] != target_len:
        # 线性更改越界结尾
        x1 = x[-2]
        y1 = orig_list[-2]
        x2 = x[-1]
        y2 = orig_list[-1]
        y_resa = (y2 - y1) * (target_len - x1) / (x2 - x1) + y1
        x[-1] = target_len
        orig_list[-1] = y_resa
    # 使用了线性的插值方法，也可以根据需要改成其他的。推荐是线性
    f = interpolate.interp1d(x, orig_list, 'linear')
    del x
    resample_list = f([x for x in range(target_len)])
    return resample_list


def find_nearest_idx(arr, value):
    arr = np.asarray(arr)
    array = abs(np.asarray(arr) - value)
    idx = array.argmin()
    return idx


def emg_rectification(x, Fs=1000, code=None, people=None):
    # Fs 采样频率，在EMG信号中是1000Hz
    x_mean = np.mean(x)
    raw = x - x_mean * np.ones_like(x)
    t = np.arange(0, raw.size / Fs, 1 / Fs)
    EMGFWR = abs(raw)

    # 线性包络 Linear Envelope
    NUMPASSES = 3
    LOWPASSRATE = 6  # 低通滤波4—10Hz得到包络线

    Wn = LOWPASSRATE / (Fs / 2)
    [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
    EMGLE = scipy.signal.filtfilt(b, a, EMGFWR, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波

    # if code == 'BIC':
    #     ref = 63.4452
    # elif code == 'BRA':
    #     ref = 29.9301
    # elif code == 'BRD':
    #     ref = 116.8316
    # elif code == 'TRI':
    #     ref = 42.5301
    # else:
    #     ref = max(EMGLE)
    # if people == 'chenzui':
    #     if code == 'BIC':
    #         ref = 407.11
    #     elif code == 'BRA':
    #         ref = 250.37
    #     elif code == 'BRD':
    #         ref = 468.31
    #     elif code == 'TRI':
    #         ref = 467.59
    #     else:
    #         ref = max(EMGLE)
    if people == 'chenzui':
        if code == 'BIC':
            ref = 703.47
        elif code == 'BRA':
            ref = 250.37
        elif code == 'BRD':
            ref = 468.31
        elif code == 'TRI':
            ref = 467.59
        else:
            ref = max(EMGLE)
    elif people == 'zhuo':
        if code == 'BIC':
            ref = 417.90
        elif code == 'BRA':
            ref = 126.73
        elif code == 'BRD':
            ref = 408.73
        elif code == 'TRI':
            ref = 250.28
        else:
            ref = max(EMGLE)
    else:
        print('No information of this people.')
    # ref = 1

    normalized_EMG = EMGLE / ref
    y = normalized_EMG
    return [y, t]
