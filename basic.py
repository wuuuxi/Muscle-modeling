import numpy as np
import scipy


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
    f = scipy.interpolate.interp1d(x, orig_list, 'linear')
    del x
    resample_list = f([x for x in range(target_len)])
    return list(resample_list)


def find_nearest_idx(arr, value):
    arr = np.asarray(arr)
    array = abs(np.asarray(arr) - value)
    idx = array.argmin()
    return idx