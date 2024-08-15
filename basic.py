import scipy.interpolate as interpolate
import numpy as np
import scipy
import pandas as pd
import os
from require import *

if sport_label == 'biceps_curl':
    if muscle_number == 3:
        if left_or_right == 'left':
            muscle_idx = ['bic_s_l', 'brachialis_1_l', 'brachiorad_1_l']
        else:
            assert left_or_right == 'right'
            muscle_idx = ['bic_s', 'brachialis_1', 'brachiorad_1']
        iso = [173, 314, 30]
        OptimalFiberLength = [0.10740956509395153, 0.0805163842230218, 0.15631665675596404]
        MaximumPennationAngle = [1.4706289056333368, 1.4706289056333368, 1.4706289056333368]
        PennationAngleAtOptimal = [0, 0, 0]
        KshapeActive = [0.45, 0.45, 0.45]
    elif muscle_number == 4:
        if left_or_right == 'left':
            muscle_idx = ['bic_s_l', 'brachialis_1_l', 'brachiorad_1_l', 'tric_long_1_l']
        else:
            assert left_or_right == 'right'
            muscle_idx = ['bic_s', 'brachialis_1', 'brachiorad_1', 'tric_long_1']
        iso = [173, 314, 30, 223]
        OptimalFiberLength = [0.10740956509395153, 0.0805163842230218, 0.15631665675596404, 0.09027865826431776]
        MaximumPennationAngle = [1.4706289056333368, 1.4706289056333368, 1.4706289056333368, 1.4706289056333368]
        PennationAngleAtOptimal = [0, 0, 0, 0.174532925199]
        KshapeActive = [0.45, 0.45, 0.45, 0.45]
    elif muscle_number == 7:
        muscle_idx = ['BICsht', 'BRA', 'BRD', 'BIClon', 'TRIlon', 'TRIlat', 'TRImed']
        iso = [316.8, 1177.37, 276.0, 525.1, 771.8, 717.5, 717.5]
elif sport_label == 'bench_press':
    joint_num = {'shoulder': '1',
                 'elbow': '2'}
    musc_label_all = ['PMCla', 'PMSte', 'PMCos', 'DelAnt', 'DelMed',
                      'DelPos', 'BicLong', 'BicSho', 'TriLong', 'TriLat',
                      'BRA', 'BRD', 'LD', 'TerMaj', 'TerMin',
                      'Infra', 'Supra', 'Cora']
    measured_muscle_idx_all = ['pect_maj_c_1', 'pect_maj_t_5', 'pect_maj_t_1', 'delt_clav_1', 'delt_scap_5',
                               'delt_scap_1', 'bic_l', 'bic_s', 'tric_long_1', 'tric_lat_1',
                               'brachialis_1', 'brachiorad_1', 'LD_T10_l', 'ter_maj_1_l', 'ter_min_1_l',
                               'infra_1', 'supra_1', 'coracobr_1']
    iso_all = [179, 133, 125, 128, 120, 289, 347, 173, 223, 153, 314, 30, 27.6, 165, 172, 272, 120, 201]
    if joint_idx == 'all':
        if muscle_number == 6:
            if left_or_right == 'left':
                muscle_idx = ['bic_s_l', 'tric_long_1_l', 'delt_clav_1_l', 'delt_scap_9_l', 'pect_maj_t_1_l',
                              'LD_T10_l']
                iso = [173.0, 223.0, 128.0, 286.0, 125.0, 27.6]
            else:
                assert left_or_right == 'right'
                muscle_idx = ['bic_s', 'tric_long_1', 'delt_clav_1', 'ter_maj_1', 'pect_maj_t_1', 'LD_T10_r']
                iso = [173.0, 223.0, 128.0, 165.0, 125.0, 27.6]
            iso_min = np.asarray(iso) * 0.1
            iso_max = np.asarray(iso) * 10
            measured_muscle_idx = muscle_idx
            musc_label = ['Biceps', 'Triceps', 'Deltoid', 'Medius', 'Pectoralis', 'Latissimus']
            related_muscle_num = [0] * (len(measured_muscle_idx))
            related_muscle_idx = []
            running_sum = len(measured_muscle_idx)
            for num in related_muscle_num:
                related_muscle_idx.append(running_sum)
                running_sum += num
        elif muscle_number == 66:
            if left_or_right == 'left':
                measured_muscle_idx = ['bic_s_l', 'tric_long_1_l', 'delt_clav_1_l', 'delt_scap_9_l', 'pect_maj_t_1_l',
                                       'LD_T10_l']
                muscle_idx = ['bic_s_l', 'tric_long_1_l', 'delt_clav_1_l', 'delt_scap_9_l', 'pect_maj_t_1_l',
                              'LD_T10_l',
                              'bic_l_l', 'brachialis_1_l', 'brachialis_2_l', 'brachialis_3_l', 'brachialis_4_l',
                              'brachialis_5_l',
                              'brachialis_6_l', 'brachialis_7_l', 'brachiorad_1_l', 'brachiorad_2_l', 'brachiorad_3_l',
                              'tric_long_2_l', 'tric_long_3_l', 'tric_long_4_l',
                              'delt_clav_2_l', 'delt_clav_3_l', 'delt_clav_4_l',
                              'delt_scap_4_l', 'delt_scap_5_l', 'delt_scap_6_l', 'delt_scap_7_l', 'delt_scap_8_l',
                              'delt_scap10_l', 'delt_scap11_l',
                              'pect_maj_t_2_l', 'pect_maj_t_3_l', 'pect_maj_t_4_l', 'pect_maj_t_5_l', 'pect_maj_t_6_l',
                              'LD_T7_l', 'LD_T8_l', 'LD_T9_l', 'LD_T11_l', 'LD_T12_l']
            else:
                measured_muscle_idx = ['bic_s', 'tric_long_1', 'delt_clav_1', 'delt_scap_9', 'pect_maj_t_1', 'LD_T10_r']
                muscle_idx = ['bic_s', 'tric_long_1', 'delt_clav_1', 'delt_scap_9', 'pect_maj_t_1', 'LD_T10_r',
                              'bic_l', 'brachialis_1', 'brachialis_2', 'brachialis_3', 'brachialis_4', 'brachialis_5',
                              'brachialis_6', 'brachialis_7', 'brachiorad_1', 'brachiorad_2', 'brachiorad_3',
                              'tric_long_2', 'tric_long_3', 'tric_long_4',
                              'delt_clav_2', 'delt_clav_3', 'delt_clav_4',
                              'delt_scap_4', 'delt_scap_5', 'delt_scap_6', 'delt_scap_7', 'delt_scap_8',
                              'delt_scap10', 'delt_scap11',
                              'pect_maj_t_2', 'pect_maj_t_3', 'pect_maj_t_4', 'pect_maj_t_5', 'pect_maj_t_6',
                              'LD_T7_r', 'LD_T8_r', 'LD_T9_r', 'LD_T11_r', 'LD_T12_r']
            iso = [173.0, 223.0, 128.0, 286.0, 125.0, 27.6,
                   173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0,
                   223.0, 223.0, 223.0,
                   128.0, 128.0, 128.0,
                   286.0, 286.0, 286.0, 286.0, 286.0, 286.0, 286.0,
                   125.0, 125.0, 125.0, 125.0, 125.0,
                   27.6, 27.6, 27.6, 27.6, 27.6]
            iso_min = np.asarray(iso) * 0.1
            iso_max = np.asarray(iso) * 10
            musc_label = ['Biceps', 'Triceps', 'Deltoid', 'Medius', 'Pectoralis', 'Latissimus']
            related_muscle_num = [11, 3, 3, 7, 5, 5]
            related_muscle_idx = []
            running_sum = len(measured_muscle_idx)
            for num in related_muscle_num:
                related_muscle_idx.append(running_sum)
                running_sum += num
        else:
            # musc_label = ['PMCla', 'PMSte', 'PMCos', 'DelAnt', 'DelMed',
            #               'DelPos', 'BicLong', 'BicSho', 'TriLong', 'TriLat',
            #               'BRA', 'BRD', 'LD']
            # muscle_idx = ['pect_maj_c_1', 'pect_maj_t_5', 'pect_maj_t_1', 'delt_clav_1', 'delt_scap_5',
            #               'delt_scap_1', 'bic_l', 'bic_s', 'tric_long_1', 'tric_lat_1',
            #               'brachialis_1', 'brachiorad_1', 'LD_T10_l', 'ter_maj_1_l', 'ter_min_1_l',
            #               'infra_1', 'supra_1', 'coracobr_1',
            #               'pect_maj_c_2',
            #               'pect_maj_t_6',
            #               'pect_maj_t_2', 'pect_maj_t_3', 'pect_maj_t_4',
            #               'delt_clav_2', 'delt_clav_3', 'delt_clav_4',
            #               'delt_scap_7', 'delt_scap_8', 'delt_scap_9', 'delt_scap_10', 'delt_scap_11',
            #               'delt_scap_2', 'delt_scap_3', 'delt_scap_4', 'delt_scap_5',
            #               'tric_long_2', 'tric_long_3', 'tric_long_4',
            #               'tric_lat_2', 'tric_lat_3', 'tric_lat_4', 'tric_lat_5',
            #               'brachialis_2', 'brachialis_3', 'brachialis_4', 'brachialis_5', 'brachialis_6',
            #               'brachiorad_2', 'brachiorad_3',
            #               'ter_maj_2_l', 'ter_maj_3_l', 'ter_maj_4_l',
            #               'ter_min_2_l', 'ter_min_3_l',
            #               'infra_2', 'infra_3', 'infra_4', 'infra_5', 'infra_6',
            #               'supra_2', 'supra_3', 'supra_4',
            #               'coracobr_2', 'coracobr_3']
            measured_muscle_idx = []
            iso = []
            for m in musc_label:
                i = musc_label_all.index(m)
                measured_muscle_idx.append(measured_muscle_idx_all[i])
                iso.append(iso_all[i])
            muscle_idx = measured_muscle_idx
            iso_min = np.asarray(iso) * 0.1
            iso_max = np.asarray(iso) * 100
            related_muscle_num = [0] * (len(measured_muscle_idx))
            related_muscle_idx = []
            running_sum = len(measured_muscle_idx)
            for num in related_muscle_num:
                related_muscle_idx.append(running_sum)
                running_sum += num
    elif joint_idx == 'shoulder':
        # musc_label = ['PMCla', 'PMSte', 'PMCos', 'DelAnt', 'DelMed', 'DelPos', 'BicLong', 'BicSho', 'TriLong',
        #               'LD', 'TerMaj', 'TerMin', 'Infra', 'Supra', 'Cora']
        musc_label = ['PMCla', 'PMSte', 'PMCos', 'DelAnt', 'DelMed', 'DelPos', 'BicLong', 'BicSho', 'TriLong',
                      'LD', 'Cora']
        measured_muscle_idx = []
        iso = []
        for m in musc_label:
            i = musc_label_all.index(m)
            measured_muscle_idx.append(measured_muscle_idx_all[i])
            iso.append(iso_all[i])
        muscle_idx = measured_muscle_idx
        iso_min = np.asarray(iso) * 0.01
        iso_max = np.asarray(iso) * 100
        related_muscle_num = [0] * (len(measured_muscle_idx))
        related_muscle_idx = []
        running_sum = len(measured_muscle_idx)
        for num in related_muscle_num:
            related_muscle_idx.append(running_sum)
            running_sum += num
    elif joint_idx == 'elbow':
        musc_label = ['BicLong', 'BicSho', 'TriLong', 'TriLat', 'BRA', 'BRD']
        measured_muscle_idx = []
        iso = []
        for m in musc_label:
            i = musc_label_all.index(m)
            measured_muscle_idx.append(measured_muscle_idx_all[i])
            iso.append(iso_all[i])
        muscle_idx = measured_muscle_idx
        iso_min = np.asarray(iso) * 0.01
        iso_max = np.asarray(iso) * 100
        related_muscle_num = [0] * (len(measured_muscle_idx))
        related_muscle_idx = []
        running_sum = len(measured_muscle_idx)
        for num in related_muscle_num:
            related_muscle_idx.append(running_sum)
            running_sum += num


elif sport_label == 'deadlift':
    joint_num = {'waist': '1',
                 'hip': '2',
                 'knee': '3'}
    musc_label = ['TA', 'GasLat', 'GasMed', 'VL', 'RF', 'VM', 'TFL', 'AddLong', 'Sem', 'BF',
                  'GMax', 'GMed', 'PsoMaj', 'IO', 'RA', 'ESI', 'Mul', 'EO', 'ESL', 'LD']
    musc_mvc = [0.1747, 0.1369, 0.1204, 0.1607, 0.1478, 0.1202, 0.0728, 0.1037, 0.2073, 0.1785,
                0.0406, 0.0511, 0.0838, 0.1731, 0.1259, 0.1388, 0.0839, 0.1107, 0.1218, 0.0754]
    if joint_idx == 'all':
        if muscle_number == 20:
            measured_muscle_idx = ['tibant_r', 'gaslat_r', 'gasmed_r', 'vaslat_r', 'recfem_r',
                                   'vasmed_r', 'tfl_r', 'addlong_r', 'semiten_r', 'bflh_r',
                                   'glmax1_r', 'glmed1_r', 'Ps_L1_VB_r', 'IO1_r', 'rect_abd_r',
                                   'IL_L1_r', 'MF_m1s_r', 'EO1_r', 'LTpT_T2_r', 'LD_L2_r']
            muscle_idx = measured_muscle_idx
            iso = [5] * 20
            related_muscle_num = [0] * 9 + [1, 2, 2, 21, 5, 1, 23, 49, 11, 39, 9]
            related_muscle_idx = [20] * 9 + [20, 21, 23, 25, 46, 51, 52, 75, 124, 135, 174]
        else:
            measured_muscle_idx = ['tibant_r', 'gaslat_r', 'gasmed_r', 'vaslat_r', 'recfem_r',
                                   'vasmed_r', 'tfl_r', 'addlong_r', 'semiten_r', 'bflh_r',
                                   'glmax1_r', 'glmed1_r', 'Ps_L1_VB_r', 'IO1_r', 'rect_abd_r',
                                   'IL_L1_r', 'MF_m1s_r', 'EO1_r', 'LTpT_T2_r']
            muscle_idx = ['tibant_r', 'gaslat_r', 'gasmed_r', 'vaslat_r', 'recfem_r',
                          'vasmed_r', 'tfl_r', 'addlong_r', 'semiten_r', 'bflh_r',
                          'glmax1_r', 'glmed1_r', 'Ps_L1_TP_r', 'IO1_r', 'rect_abd_r',
                          'IL_R5_r', 'MF_m1t_3_r', 'EO1_r', 'LTpT_T2_r',
                          'addbrev_r', 'semimem_r', 'bfsh_r',
                          'glmax2_r', 'glmax3_r',
                          'glmed2_r', 'glmed3_r',
                          'Ps_L2_TP_r', 'Ps_L3_TP_r', 'Ps_L4_TP_r', 'Ps_L5_TP_r', 'Ps_L1_VB_l', 'Ps_L2_TP_l',
                          'Ps_L3_TP_l', 'Ps_L4_TP_l', 'Ps_L5_TP_l',
                          'IO1_l', 'rect_abd_l', 'IL_R5_l', 'MF_m1t_3_l', 'EO1_l', 'LTpT_T2_l'
                          ]
            iso = [1227, 1575, 3115, 5148, 2192,
                   2748, 411, 917, 591, 1313,
                   983, 1093, 161, 124, 163,
                   21, 27, 111, 50,
                   626, 2200, 557,
                   1406, 948,
                   765, 871,
                   160, 77, 122, 132, 161, 160, 77, 122, 132,
                   124, 163, 21, 27, 111, 50]
            iso_min = np.asarray(iso) * 0.1
            iso_max = np.asarray(iso) * 10
            iso_max[15:19] = np.asarray([1060, 490, 2000, 1000])
            related_muscle_num = [0] * (len(measured_muscle_idx) - 12) + [1, 1, 1, 2, 2, 9, 1, 1, 1, 1, 1, 1]
            related_muscle_idx = []
            running_sum = len(measured_muscle_idx)
            for num in related_muscle_num:
                related_muscle_idx.append(running_sum)
                running_sum += num
    elif joint_idx == 'waist':
        if muscle_number == 24:
            measured_muscle_idx = ['IO1_r', 'rect_abd_r', 'IL_R5_r', 'MF_m1t_3_r', 'EO1_r', 'LTpT_T2_r']
            muscle_idx = ['IO1_r', 'rect_abd_r', 'IL_R5_r', 'MF_m1t_3_r', 'EO1_r', 'LTpT_T2_r',
                          'IO1_l', 'rect_abd_l', 'IL_R5_l', 'MF_m1t_3_l',
                          'EO1_l', 'EO4_l', 'EO6_l', 'EO4_r', 'EO6_r',
                          'LTpT_T12_l', 'LTpT_R4_l', 'LTpT_R12_l', 'LTpL_L5_l', 'LTpT_T2_l',
                          'LTpT_T12_r', 'LTpT_R4_r', 'LTpT_R12_r', 'LTpL_L5_r']
            magn = 10  # default magnification
            iso_min = np.asarray([124, 163, 21, 27, 111, 50,
                                  124, 163, 21, 27,
                                  112, 134, 227, 134, 227,
                                  62, 20, 62, 103, 50, 62, 20, 62, 103]) * 0.5
            iso_max = np.asarray([124 * magn, 163 * magn, 1060, 490, 112 * magn, 50 * magn,
                                  124 * magn, 163 * magn, 1060, 490,
                                  112 * magn, 134 * magn, 227 * magn, 134 * magn, 227 * magn,
                                  62 * magn, 20 * magn, 62 * magn, 103 * magn, 50 * magn, 62 * magn, 20 * magn,
                                  62 * magn,
                                  103 * magn])
            musc_label = ['IO', 'RA', 'ESI', 'Mul', 'EO', 'ESL']
            related_muscle_num = [1] * (len(measured_muscle_idx) - 2) + [5, 9]
        elif muscle_number == 30:
            measured_muscle_idx = ['Ps_L1_VB_r', 'IO1_r', 'rect_abd_r', 'IL_R5_r', 'MF_m1t_3_r', 'EO1_r',
                                   'LTpT_T2_r', 'LD_L2_r']
            muscle_idx = ['Ps_L1_VB_r', 'IO1_r', 'rect_abd_r', 'IL_R5_r', 'MF_m1t_3_r', 'EO1_r', 'LTpT_T2_r', 'LD_L2_r',
                          'Ps_L1_VB_l', 'IO1_l', 'rect_abd_l', 'IL_R5_l', 'MF_m1t_3_l',
                          'EO1_l', 'EO4_l', 'EO6_l', 'EO4_r', 'EO6_r',
                          'LTpT_T12_l', 'LTpT_R4_l', 'LTpT_R12_l', 'LTpL_L5_l', 'LTpT_T2_l',
                          'LTpT_T12_r', 'LTpT_R4_r', 'LTpT_R12_r', 'LTpL_L5_r',
                          'LD_L2_l', 'LD_Il_l', 'LD_Il_r']
            iso = [5] * len(muscle_idx)
            musc_label = ['PsoMaj', 'IO', 'RA', 'ESI', 'Mul', 'EO', 'ESL', 'LD']
            related_muscle_num = [1] * (len(measured_muscle_idx) - 3) + [5, 9, 3]
        else:
            measured_muscle_idx = ['Ps_L1_TP_r', 'IO1_r', 'rect_abd_r', 'IL_R5_r', 'MF_m1t_3_r', 'EO1_r', 'LTpT_T2_r']
            muscle_idx = ['Ps_L1_TP_r', 'IO1_r', 'rect_abd_r', 'IL_R5_r', 'MF_m1t_3_r', 'EO1_r', 'LTpT_T2_r',
                          'Ps_L1_TP_r', 'IO1_l', 'rect_abd_l', 'IL_R5_l', 'MF_m1t_3_l', 'EO1_l', 'LTpT_T2_l']
            magn = 10  # default magnification
            iso_min = np.asarray([161, 124, 163, 21, 27, 111, 50,
                                  161, 124, 163, 21, 27, 111, 50]) * 0.5
            iso_max = np.asarray([161 * magn, 124 * magn, 163 * magn, 1060, 490, 112 * magn, 50 * magn,
                                  161 * magn, 124 * magn, 163 * magn, 1060, 490, 112 * magn, 50 * magn])
            # musc_label = ['IO', 'RA', 'ESI', 'Mul', 'EO', 'ESL']
            musc_label = musc_label[12:19]
            # related_muscle_num = [1] * (len(measured_muscle_idx) - 2) + [5, 9]
            related_muscle_num = [1] * (len(measured_muscle_idx))
        related_muscle_idx = []
        running_sum = len(measured_muscle_idx)
        for num in related_muscle_num:
            related_muscle_idx.append(running_sum)
            running_sum += num
    elif joint_idx == 'hip':
        measured_muscle_idx = ['recfem_r', 'tfl_r', 'addlong_r', 'semiten_r', 'bflh_r',
                               'glmax1_r', 'glmed1_r', 'Ps_L1_TP_r']
        muscle_idx = ['recfem_r', 'tfl_r', 'addlong_r', 'semiten_r', 'bflh_r',
                      'glmax1_r', 'glmed1_r', 'Ps_L1_TP_r',
                      'addbrev_r', 'semimem_r', 'bfsh_r', 'glmax2_r', 'glmax3_r', 'glmed2_r', 'glmed3_r',
                      'Ps_L2_TP_r', 'Ps_L3_TP_r', 'Ps_L4_TP_r', 'Ps_L5_TP_r']
        iso = [2192, 411, 917, 591, 1313, 983, 1093, 161,
               626, 2200, 557, 1406, 948, 765, 871, 160, 77, 122, 132]
        iso_min = np.asarray(iso) * 0.5
        iso_max = np.asarray(iso) * 10
        musc_label = ['RF', 'TFL', 'AddLong', 'Sem', 'BF', 'GMax', 'GMed', 'PsoMaj']
        # musc_label = musc_label
        related_muscle_num = [0] * (len(measured_muscle_idx) - 6) + [1, 1, 1, 2, 2, 4]
        related_muscle_idx = []
        running_sum = len(measured_muscle_idx)
        for num in related_muscle_num:
            related_muscle_idx.append(running_sum)
            running_sum += num
    elif joint_idx == 'knee':
        measured_muscle_idx = ['tibant_r', 'gaslat_r', 'gasmed_r', 'vaslat_r', 'recfem_r',
                               'vasmed_r', 'semiten_r', 'bflh_r']
        muscle_idx = ['tibant_r', 'gaslat_r', 'gasmed_r', 'vaslat_r', 'recfem_r',
                      'vasmed_r', 'semiten_r', 'bflh_r']
        iso = [1227, 1575, 3115, 5148, 2192, 2748, 591, 1313]
        iso_min = np.asarray(iso) * 0.5
        iso_max = np.asarray(iso) * 10
        musc_label = ['TA', 'GasLat', 'GasMed', 'VL', 'RF', 'VM', 'Sem', 'BF']
        related_muscle_num = [0] * (len(measured_muscle_idx))
        related_muscle_idx = []
        running_sum = len(measured_muscle_idx)
        for num in related_muscle_num:
            related_muscle_idx.append(running_sum)
            running_sum += num


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


def emg_rectification(x, Fs=1000, code=None, people=None, left=False):
    nyquist_frequency = Fs / 2
    low_cutoff = 20 / nyquist_frequency
    high_cutoff = 450 / nyquist_frequency
    [b, a] = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band', analog=False)
    x = scipy.signal.filtfilt(b, a, x)

    # Fs 采样频率，在港大的EMG信号中是1000Hz
    x_mean = np.mean(x)
    raw = x - x_mean * np.ones_like(x)
    t = np.arange(0, raw.size / Fs, 1 / Fs)
    EMGFWR = abs(raw)

    # 线性包络 Linear Envelope
    NUMPASSES = 3
    LOWPASSRATE = 5  # 低通滤波4—10Hz得到包络线

    Wn = LOWPASSRATE / (Fs / 2)
    [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
    EMGLE = scipy.signal.filtfilt(b, a, EMGFWR, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波

    if people == 'chenzui':  # left
        if code == 'BIC':
            ref = 407.11
        elif code == 'BRA':
            ref = 250.37
        elif code == 'BRD':
            ref = 468.31
        elif code == 'TRI':
            ref = 467.59
        elif code == 'ANT':
            ref = 176.04
        elif code == 'POS':
            ref = 272.30
        elif code == 'PEC':
            ref = 87.11
        elif code == 'LAT':
            ref = 339.21
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
        elif code == 'ANT':
            ref = 280.711
        elif code == 'POS':
            ref = 116.263
        elif code == 'PEC':
            ref = 223.172
        elif code == 'LAT':
            ref = 108.985 * 10
        else:
            ref = max(EMGLE)
    elif people == 'kehan':
        if left is False:
            benchpress_musc_label = ['PMCla', 'PMSte', 'PMCos', 'DelAnt', 'DelMed', 'DelPos',
                                     'BicLong', 'BicSho', 'TriLong', 'TriLat', 'BRA', 'BRD',
                                     'LD', 'TerMaj', 'TerMin', 'Infra', 'Supra', 'Cora']
            benchpress_mvc = [0.116, 0.078, 0.131, 0.1738, 0.3129, 0.3751, 0.0719, 0.1922, 0.2832, 0.2557, 0.2298,
                              0.2906, 0.0927, 0.2927, 0.1766, 0.2214, 0.0837, 0.1591, ]
            if code in benchpress_musc_label:
                index = benchpress_musc_label.index(code)
                ref = benchpress_mvc[index]
            else:
                ref = 1
                print(f'No muscle \'{code}\' of people \'{people}\', used {ref} as MVC')
    elif people == 'yuetian':  # left
        if left is True:
            if code == 'LTA':
                ref = 3.989497
            elif code == 'LGL':
                ref = 3.319789
            elif code == 'LVL':
                ref = 4.036685
            elif code == 'LRF':
                ref = 3.125095
            elif code == 'LST':
                ref = 3.45323
            elif code == 'LBF':
                ref = 4.025123
            else:
                ref = max(EMGLE)
        else:
            benchpress_musc_label = ['PMCla', 'PMSte', 'PMCos', 'DelAnt', 'DelMed',
                                     'DelPos', 'BicLong', 'BicSho', 'TriLong', 'TriLat',
                                     'BRA', 'BRD', 'LD', 'TerMaj', 'TerMin',
                                     'Infra', 'Supra', 'Cora']
            benchpress_mvc = [0.1276, 0.0980, 0.1523, 0.3392, 0.1792, 0.3061, 0.7502, 0.9241, 0.2260, 0.3334,
                              0.2839, 0.3180, 0.1129, 0.1759, 0.1408, 0.2714, 0.1454, 0.9309]
            # benchpress_musc_label = ['BIC', 'TRI', 'ANT', 'POS', 'PEC', 'LAT', 'BRA', 'BRD']
            # benchpress_mvc = [0.67077, 0.67077, 0.263597, 0.23831, 0.43018, 0.15176, 0.67077, 0.67077]
            deadlift_waist_musc_label = ['PM', 'IO', 'RA', 'ESI', 'Mul', 'EO', 'ESL', 'LD']
            deadlift_waist_mvc = [0.0838 * 2, 0.1731 * 2, 0.1259 * 1.5, 0.1388, 0.0839 * 1.5, 0.1107 * 1.5, 0.1218,
                                  0.0754 * 2]
            deadlift_musc_label = ['TA', 'GasLat', 'GasMed', 'VL', 'RF', 'VM', 'TFL', 'AddLong', 'Sem', 'BF',
                                   'GMax', 'GMed', 'PsoMaj', 'IO', 'RA', 'ESI', 'Mul', 'EO', 'ESL', 'LD']
            deadlift_mvc = [0.1747, 0.1369, 0.1204, 0.1607, 0.1478, 0.1202, 0.0728, 0.1037, 0.2073, 0.1785,
                            0.0406, 0.0511, 0.0838, 0.1731, 0.1259, 0.1388, 0.0839, 0.1107, 0.1218, 0.0754]
            if code in benchpress_musc_label:
                index = benchpress_musc_label.index(code)
                ref = benchpress_mvc[index]
            elif code in deadlift_waist_musc_label:
                index = deadlift_waist_musc_label.index(code)
                ref = deadlift_waist_mvc[index]
            elif code in deadlift_musc_label:
                index = deadlift_musc_label.index(code)
                ref = deadlift_mvc[index]
            else:
                ref = 2
                print(f'No muscle \'{code}\' of people \'{people}\', used {ref} as MVC')
    else:
        ref = 1
        print(f'No information about people \'{people}\', used {ref} as MVC')
    # ref = 1

    normalized_EMG = EMGLE / ref
    y = normalized_EMG
    return [y, t]


def elbow_emg(emg1, emg2):
    # emg1: Biceps
    # emg2: Triceps
    x = np.asarray([[5.79835623, 1.40694018, 0.26443844, 1.34689293],
                    [0.17944381, 1.81618533, 4.21557155, 0.0]])
    # x = np.asarray([[5.02488458, 0.67736425, 1.00691882, 0.47668198],
    #                 [0.0, 2.30828341, 0.47426132, 2.30039555]])
    emg1 = np.asarray(emg1)
    emg2 = np.asarray(emg2)
    s1 = -(x[1, 2] * emg2 - x[1, 3] * emg1) / (x[0, 2] * x[1, 3] - x[1, 2] * x[0, 3])
    s2 = (x[0, 2] * emg2 - x[0, 3] * emg1) / (x[0, 2] * x[1, 3] - x[1, 2] * x[0, 3])
    a1 = s1 * x[0, 0] + s2 * x[1, 0]
    a2 = s1 * x[0, 1] + s2 * x[1, 1]
    return a1, a2


def from_csv_to_xlsx(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if 'csv' in file:
                xlsx_file = file[:-4] + '.xlsx'
                # 由于处理HKU的emg，从第6行开始读
                # data_frame = pd.read_csv(root + '/' + file, skiprows=5)
                data_frame = pd.read_csv(root + '/' + file)
                data_frame.to_excel(root + '/' + xlsx_file, index=False)


def from_csv_to_emg(csv_name):
    states = pd.read_csv(csv_name, low_memory=False, skiprows=2)
    emg = np.squeeze(np.asarray([states]))
    emg = np.concatenate([emg[3:, 0:13], emg[3:, 19:25]], axis=1).astype(float)  # 前12和后6, 包括第一列时间
    # emg = np.concatenate([emg[3:, 1:13], emg[3:, 19:25]], axis=1).astype(float)
    # emg = emg[3:, 1:19].astype(float)
    return emg


if __name__ == '__main__':
    # from_csv_to_xlsx('files/bench press/kehan/Kehan bench')
    from_csv_to_xlsx('files/bench press/yuetian/0408/20240408健身机数据')
