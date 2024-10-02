# muscle_number = 6
muscle_number = 19
left_or_right = 'right'
# left_or_right = 'left'
left = False
# sport_label = 'biceps_curl'
# sport_label = 'bench_press'
# sport_label = 'deadlift'
sport_label = 'squat'
joint_idx = 'all'
date = '240704'

iso1 = [100, 100, 10]
iso2 = [6000, 6000, 3000]

emg_lift_len = 1000
target_len = 50
fs = 1000
# include_state = 'lift'
# include_state = 'down'
include_state = 'lift and down'
# include_down = True
# only_lift = False
include_TRI = False
torque_init_0 = False

mvc_is_variable = False
plot_distribution = True
arm_constant = False
# arm_constant = True
need_all_muscle = True
legend_label = True
elbow_muscle = True

muscle_LAT = True
seven_six = True
activation_label = True

# musc_label_all = ['PMCla', 'PMSte', 'PMCos', 'DelAnt', 'DelMed', 'DelPos',
#                   'BicLong', 'BicSho', 'TriLong', 'TriLat', 'BRA', 'BRD',
#                   'LD', 'TerMaj', 'TerMin', 'Infra', 'Supra', 'Cora']
#
musc_label = ['PMCla', 'PMSte', 'PMCos', 'DelAnt', 'DelMed', 'DelPos',
              'BicLong', 'BicSho', 'TriLong', 'TriLat', 'BRA', 'BRD',
              'LD']

# musc_label = ['PMCla', 'PMSte', 'PMCos', 'DelAnt', 'DelMed', 'DelPos',
#               'BicLong', 'BicSho', 'TriLong', 'TriLat', 'BRA', 'BRD',
#               'LD', 'TerMaj', 'TerMin', 'Infra', 'Supra', 'Cora']
