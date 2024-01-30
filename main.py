from emg_distribution import *
from test import *
from require import *


def optimization(emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d):
    m = gekko.GEKKO(remote=False)
    x = m.Array(m.Var, arm.shape, lb=0, ub=1)  # activation
    y = m.Array(m.Var, len(muscle_idx), lb=0)  # maximum isometric force
    c = m.Array(m.Var, len(muscle_idx))  # MVC emg
    f = m.Array(m.Var, arm.shape)  # muscle force
    t = m.Array(m.Var, torque.shape)  # torque
    # m.Minimize(np.square(x - emg).mean())
    m.Minimize(np.square(t - torque).mean())
    for i in range(arm.shape[0]):
        for j in range(arm.shape[1]):
            m.Equation(x[i][j] * y[i] == f[i, j])
            # m.Equation(x[i][j] * y[i] * fa[i][j] == f[i, j])
            m.Equation(sum(f[:, j] * arm[:, j]) == t[j])
            if mvc_is_variable is True:
                m.Equation(x[i][j] <= (emg_mean[i][j] * c[i] + emg_std[i][j] * c[i] * 2))
                m.Equation(x[i][j] >= (emg_mean[i][j] * c[i] - emg_std[i][j] * c[i] * 2))
                m.Equation((emg_mean[i][j] * c[i] + emg_std[i][j] * c[i] * 2) <= 1)
                # m.Equation((emg_mean[i][j] * c[i] - emg_std[i][j] * c[i] * 2) >= 0)
            else:
                m.Equation(x[i][j] <= emg[i][j] * 1.20)
                m.Equation(x[i][j] >= emg[i][j] * 0.80)
                # m.Equation(x[i][j] <= (emg_mean[i][j] + emg_std[i][j] * 2))
                # m.Equation(x[i][j] >= (emg_mean[i][j] - emg_std[i][j] * 2))
        # m.Equations([y[i] >= 3 * iso1[i], y[i] <= 3 * iso2[i]])
        # m.Equations([y[i] >= iso1[i], y[i] <= iso2[i]])
        m.Equations([y[i] >= 0.5 * iso[i], y[i] <= 100 * iso[i]])
        # m.Equations([y[i] >= 0.5 * iso[i], y[i] <= 3 * iso[i]])
        if mvc_is_variable is True:
            m.Equations([c[i] >= 0.01, c[i] <= 5])
    # m.Equations([y[0] >= 0.5 * y[1], y[0] <= 2.0 * y[1]])
    # m.Equations([y[2] >= 0.1 * y[0], y[2] >= 0.1 * y[1]])
    # m.options.MAX_ITER = 10000
    m.solve(disp=False)
    # print(x)
    # print(y)
    for i in range(y.size):
        print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(y)[i].value[0]))
    if mvc_is_variable is True:
        print('-' * 50)
        for i in range(c.size):
            print(muscle_idx[i], ':\t', "{:.2f}".format(np.asarray(c)[i].value[0]))

    r = np.ones_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            r[i][j] = r[i][j] * x[i][j].value[0]
    e = np.ones_like(t)
    for i in range(t.shape[0]):
        e[i] = e[i] * t[i].value[0]
    t = e

    c_r = [(np.asarray(c)[i].value[0]) for i in range(c.size)]
    y_r = [(np.asarray(y)[i].value[0]) for i in range(y.size)]  # maximum isometric force
    if mvc_is_variable is True:
        active_force = emg.T * c_r * y_r
    else:
        active_force = emg.T * y_r
        # active_force = fa.T * emg.T * y_r
    calu_torque = [sum(active_force[j, :] * arm[:, j]) for j in range(arm.shape[1])]
    return t, r, calu_torque, c, y_r


def one_repetition(label, idx):
    for i in range(len(idx)):
        print('-' * 25, i, '-' * 25)
        emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
            = read_realted_files(label, idx[i])
        t, r, calu_torque, c, y_r = optimization(emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg,
                                                 time_emg, emg_trend_u, emg_trend_d)
        rmse = np.sqrt(np.sum((np.asarray(t) - torque) ** 2) / len(torque))
        print("torque rmse:\t", "{:.2f}".format(rmse))
        plot_result(0, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c, i)


if __name__ == '__main__':
    calculate_chenzui_emg_distribution(label='6.5kg-cts')
    # calculate_lizhuo_emg_distribution(label='1kg')
    # plt.show()

    # label = 'chenzui-left-all-6.5kg-cts'
    # idx = ['23-1', '23-2', '23-3', '23-4', '23-5']
    # label = 'chenzui-left-all-4kg-cts'
    # idx = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    # label = 'chenzui-left-all-5.5kg-cts'
    # idx = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    label = 'chenzui-left-all-6.5kg-cts'
    idx = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    one_repetition(label, idx)

    # emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
    #     = read_groups_files('chenzui-left-all-3kg')
    # # emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
    # #     = read_realted_files(label='chenzui-left-all-6.5kg-cts', idx='23-1')
    #
    # t, r, calu_torque, c, y_r = optimization(emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d)
    #
    # print('-' * 25, 'training rmse', '-' * 25)
    # plot_all_result(1, t, r, emg, time, torque, emg_std_long, emg_mean_long, calu_torque, c)
    #
    # print('-' * 25, 'application', '-' * 25)
    # application('chenzui-left-all-3kg', y_r)
    # # application('chenzui-left-all-4kg', y_r)
    # # emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
    # #     = read_realted_files(label='chenzui-left-all-2.5kg', idx='8')
    # # test_optimization_emg(3, y_r, torque, time, emg_mean_long, emg_std_long, arm, emg, emg_mean, emg_std,
    # #                       emg_trend_u, emg_trend_d)
    # # emg_mean, emg_std, arm, torque, time, emg_mean_long, emg_std_long, emg, time_emg, emg_trend_u, emg_trend_d \
    # #     = read_realted_files(label='chenzui-left-all-6.5kg', idx='18')
    # # test_optimization_emg(4, y_r, torque, time, emg_mean_long, emg_std_long, arm, emg, emg_mean, emg_std,
    # #                       emg_trend_u, emg_trend_d)

    plt.show()
