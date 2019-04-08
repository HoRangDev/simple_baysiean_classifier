import copy
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

def load_datas(filePath):
    # feature_1 feature_2 feature_3 feature_4 class
    file = open(filePath, 'r')

    listOfDatas = [];
    while True:
        readCheck = file.readline()
        if(not readCheck):
            break;

        splitList = readCheck.split('\t')
        if (len(splitList) >= 2 ):
            features = splitList[0].split('   ')
            classes = splitList[1].split(' ')
            listOfDatas.append([float(features[0]), float(features[1]), float(features[2]), float(features[3]), int(classes[0])])

    file.close()

    return listOfDatas;

def print_datas(datas):
    print("#Data: " + str(len(datas)));
    for data in datas:
        print('feature 0: ' + str(data[0])
              + '\tfeature 1: ' + str(data[1])
              + '\tfeature 2: ' + str(data[2])
              + '\tfeature 3: ' + str(data [3]) 
              + '\t class: ' + str(data[4]));

def mean4D(samples, class_idx, num_of_samples):
    mean = [0, 0, 0, 0]
    for sample in range(0, num_of_samples):
        sample_data = samples[class_idx * num_of_samples + sample]
        for idx in range(0, 4):
            mean[idx] += sample_data[idx]
    for idx in range(0, 4):
        mean[idx] /= num_of_samples
    return mean

def mean2D(samples, class_idx, num_of_samples):
    mean = [0, 0]
    for sample in range(0, num_of_samples):
        sample_data = samples[class_idx * num_of_samples + sample]
        for idx in range(0, 2):
            mean[idx] += sample_data[idx]
    for idx in range(0, 2):
        mean[idx] /= num_of_samples
    return mean

def vec4_mul_transposed_vec4(left, right):
    # mat[col][row]
    mat = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for col in range(0, 4):
        for row in range(0, 4):
            mat[col][row] = left[row]*right[col]
    return mat

def vec2_mul_transposed_vec2(left, right):
    # mat[col][row]
    mat = [[0, 0], [0, 0]]
    for col in range(0, 2):
        for row in range(0, 2):
            mat[col][row] = left[row]*right[col]
    return mat

def matNbyN_scala_divide(n_dims, mat, scala):
    res_mat = copy.deepcopy(mat)
    for col in range(0, n_dims):
        for row in range(0, n_dims):
            res_mat[col][row] /= scala
    return res_mat

def matNbyN_matNbyN_plus(n_dims, mat_left, mat_right):
    res_mat = copy.deepcopy(mat_left)
    for col in range(0, n_dims):
        for row in range(0, n_dims):
            res_mat[col][row] += mat_right[col][row]
    return res_mat

def matNbyN_matNbyN_minus(n_dims, mat_left, mat_right):
    res_mat = copy.deepcopy(mat_left)
    for col in range(0, n_dims):
        for row in range(0, n_dims):
            res_mat[col][row] -= mat_right[col][row]
    return res_mat

def covariance_mat_4D(samples, class_idx, num_of_samples, mean_vector):
    result = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    mean_matrix = vec4_mul_transposed_vec4(mean_vector, mean_vector)
    for sample_idx in range(0, num_of_samples):
        sample = samples[class_idx * num_of_samples + sample_idx]
        features = [sample[0], sample[1], sample[2], sample[3]]
        feature_matrix = vec4_mul_transposed_vec4(features, features)
        result = matNbyN_matNbyN_plus(4,
                                      result,
                                      matNbyN_matNbyN_minus(4, feature_matrix, mean_matrix))
    result = matNbyN_scala_divide(4, result, num_of_samples)
    return result

def covariance_mat_2D(samples, class_idx, num_of_samples, mean_vector):
    result = [[0, 0], [0, 0]]
    mean_matrix = vec2_mul_transposed_vec2(mean_vector, mean_vector)
    for sample_idx in range(0, num_of_samples):
        sample = samples[class_idx * num_of_samples + sample_idx]
        features = [sample[0], sample[1]]
        feature_matrix = vec2_mul_transposed_vec2(features, features)
        result = matNbyN_matNbyN_plus(2, 
                                      result,
                                      matNbyN_matNbyN_minus(2, feature_matrix, mean_matrix))
    result = matNbyN_scala_divide(2, result, num_of_samples)
    return result

def transpose(n_dims, mat):
    result = copy.deepcopy(mat);
    for col in range(0, n_dims):
        for row in range(0, n_dims):
            result[row][col] = mat[col][row]
    return result

def inverse_of_4x4_mat(mat):
    target = transpose(4, mat)
    A2323 = target[2][2] * target[3][3] - target[2][3] * target[3][2]
    A1323 = target[2][1] * target[3][3] - target[2][3] * target[3][1]
    A1223 = target[2][1] * target[3][2] - target[2][2] * target[3][1]
    A0323 = target[2][0] * target[3][3] - target[2][3] * target[3][0]
    A0223 = target[2][0] * target[3][2] - target[2][2] * target[3][0]
    A0123 = target[2][0] * target[3][1] - target[2][1] * target[3][0]
    A2313 = target[1][2] * target[3][3] - target[1][3] * target[3][2]
    A1313 = target[1][1] * target[3][3] - target[1][3] * target[3][1]
    A1213 = target[1][1] * target[3][2] - target[1][2] * target[3][1]
    A2312 = target[1][2] * target[2][3] - target[1][3] * target[2][2]
    A1312 = target[1][1] * target[2][3] - target[1][3] * target[2][1]
    A1212 = target[1][1] * target[2][2] - target[1][2] * target[2][1]
    A0313 = target[1][0] * target[3][3] - target[1][3] * target[3][0]
    A0213 = target[1][0] * target[3][2] - target[1][2] * target[3][0]
    A0312 = target[1][0] * target[2][3] - target[1][3] * target[2][0]
    A0212 = target[1][0] * target[2][2] - target[1][2] * target[2][0]
    A0113 = target[1][0] * target[3][1] - target[1][1] * target[3][0]
    A0112 = target[1][0] * target[2][1] - target[1][1] * target[2][0]

    det = target[0][0] * ( target[1][1] * A2323 - target[1][2] * A1323 + target[1][3] * A1223 ) 
    - target[0][1] * ( target[1][0] * A2323 - target[1][2] * A0323 + target[1][3] * A0223 ) 
    + target[0][2] * ( target[1][0] * A1323 - target[1][1] * A0323 + target[1][3] * A0123 )
    - target[0][3]* (  target[1][0] * A1223 - target[1][1] * A0223 + target[1][2] * A0123 )
    det = 1 / det;

    result = [
        [det * (target[1][1]*A2323 -target[1][2]*A1323 + target[1][3] * A1223), 
         -det * (target[0][1] * A2323 - target[0][2] * A1323 + target[0][3] * A1223),
        det * (target[0][1] * A2313 + target[0][2] * A1313 + target[0][3] * A1213),
       -det * (target[0][1] * A2312 - target[0][2] * A1312 + target[0][3] * A1212) ],

        [-det * (target[1][0]*A2323-target[1][2]*A0323+target[1][3]*A0223), 
         det*(target[0][0]*A2323 - target[0][2]*A0323 + target[0][3]*A1212), 
         -det*(target[0][0]*A2313-target[0][2]*A0313+target[0][3]*A0223),
        det*(target[0][0]*A2312-target[0][2]*A0312+target[0][3]*A0213)],

        [det*(target[1][0] * A1323-target[1][1]*A0323+target[1][3]*A0123), 
         -det*(target[0][0]*A1323-target[0][1]*A0323+target[0][3]*A0123),
        det*(target[0][0]*A1313 - target[0][1] * A0313 + target[0][3] * A0113),
       -det * (target[0][0] * A1312 - target[0][1] * A0312 + target[0][3] * A0112)],

        [-det*(target[1][0]*A1223 - target[1][1] * A0223 + target[1][2] * A0123), 
         det*(target[0][0]*A1223 - target[0][1]*A0223+target[0][2]*A0123),
         -det*(target[0][0]*A1213-target[0][1]*A0213+target[0][2]*A0113),
         det*(target[0][0]*A1212 - target[0][1]*A0212+target[0][2]*A0112)]]

    return transpose(4, result)

def inverse_of_2x2_mat(mat):
    det = mat[0][0] * mat[1][1] - (mat[1][0] * mat[0][1])
    det = 1/det

    result = copy.deepcopy(mat)
    result[0][0] = det * mat[1][1]
    result[1][0] = det * (-mat[1][0])
    result[0][1] = det * (-mat[0][1])
    result[1][1] = det * (mat[0][0])

    return result

def matNbyN_mul_vecN(n_dims, mat, vec):
    result = [];
    for rev in range(n_dims):
        result.append(0)

    for col in range(0, n_dims):
        for row in range(0, n_dims):
            result[row] += mat[row][col]*vec[row]
    return result

def decision_func(n_dims, input, inv_covariance_mat, mean_vector, omega):
    inv_cov_mul_mean = matNbyN_mul_vecN(n_dims, inv_covariance_mat, mean_vector) # N-D 
    input_term = 0
    constant_term = 0
    log_term = math.log(omega);
    for mean_idx in range(n_dims):
        constant_term += (inv_cov_mul_mean[mean_idx] * mean_vector[mean_idx])
        input_term += (inv_cov_mul_mean[mean_idx] * input[mean_idx])
    constant_term *= (-0.5)
    return input_term + constant_term + log_term;

def print_NbyN_matrix(n_dims, mat):
    for col in range(n_dims):
        print('col ', col, ': ', mat[col])

def plot_2D_samples(samples):
    class_sample_mask_labels = [['go', 'class 1'], ['ro', 'class 2'], ['bo', 'class 3']]
    class_samples_x = [[], [], []]
    class_samples_y = [[], [], []]

    for sample in samples:
        class_of_sample = sample[4]
        class_samples_x[class_of_sample - 1].append(sample[0])
        class_samples_y[class_of_sample - 1].append(sample[1])

    plt.plot(class_samples_x[0], class_samples_y[0],
            class_sample_mask_labels[0][0], label = class_sample_mask_labels[0][1])
    plt.plot(class_samples_x[1], class_samples_y[1], 
             class_sample_mask_labels[1][0], label = class_sample_mask_labels[1][1])
    plt.plot(class_samples_x[2], class_samples_y[2], 
             class_sample_mask_labels[2][0],label =  class_sample_mask_labels[2][1])

def plot_2D_boundaries(x_range, y_range, step, epsilon, inv_cov_mat_c1, inv_cov_mat_c2, mean_vec_c1, mean_vec_c2, omega, class_idx_c1, class_idx_c2):
    presets = ['c-', 'm-', 'k-']
    decision_x_set = []
    decision_y_set = []
    x = x_range[0]
    while (x <= x_range[1]):
        y = y_range[0]
        while (y <= y_range[1]):
            decision_val = decision_func(2, [x, y], inv_cov_mat_c1, mean_vec_c1, omega) - decision_func(2, [x, y], inv_cov_mat_c2, mean_vec_c2, omega)
            #print("desc val: " , decision_val)
            if ( (decision_val >= -epsilon) and (decision_val <= epsilon) ):
                decision_x_set.append(x)
                decision_y_set.append(y)
            y+= step
        x += step
    plt.plot(decision_x_set, decision_y_set,
            presets[class_idx_c1 + class_idx_c2-1],
           label = ('class ' + str(class_idx_c1+1) + ' & class' + str(class_idx_c2+1)))

# 40 samples for training/10 samples for test/ 50 samples for each classes, and total 150 samples for 3 classes
training_data = load_datas('Iris_train.dat');
test_data = load_datas('Iris_test.dat');
print_datas(training_data)
print_datas(test_data)

# HW_1
# Colum-major matrix
print('- HW1 -')
mean_list = []
covariance_mat_list = []
inv_convraicne_mat_list = []
for class_idx in range(0, 3):
    mean_list.append(mean4D(training_data, class_idx, 40))
    covariance_mat_list.append(covariance_mat_4D(training_data, class_idx, 40, mean_list[class_idx]))
    inv_convraicne_mat_list.append(inverse_of_4x4_mat(covariance_mat_list[class_idx]))

confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
for outter_idx in range(len(test_data)):
    sample = test_data[outter_idx]
    for class_idx in range(3):
        if (sample[4] != (class_idx+1)):
            confusion_val = decision_func(4,
                                         [sample[0], sample[1], sample[2], sample[3]], 
                                         inv_convraicne_mat_list[sample[4]-1],
                                        mean_list[sample[4]-1], 0.33 )
            confusion_val -= decision_func(4,
                                           [sample[0], sample[1], sample[2], sample[3]],
                                           inv_convraicne_mat_list[class_idx],
                                           mean_list[class_idx], 0.33 )
            if (confusion_val > 0.0):
                confusion_matrix[sample[4]-1][class_idx] -= 1
            else:
                confusion_matrix[sample[4]-1][sample[4]-1] += 1

print("HW1: Confusion Matrix")
print_NbyN_matrix(3, confusion_matrix)
print('---------------------------------------------------')
# HW_2
# Colum-major matrix
print('- HW2 -')
mean_list = []
covariance_mat_list = []
inv_convraicne_mat_list = []
for class_idx in range(0, 3):
    mean_list.append(mean2D(training_data, class_idx, 40))
    covariance_mat_list.append(covariance_mat_2D(training_data, class_idx, 40, mean_list[class_idx]))
    inv_convraicne_mat_list.append(inverse_of_2x2_mat(covariance_mat_list[class_idx]))

    
confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
for outter_idx in range(len(test_data)):
    sample = test_data[outter_idx]
    for class_idx in range(3):
        if (sample[4] != (class_idx+1)):
            confusion_val = decision_func(2,
                                         [sample[0], sample[1], sample[2], sample[3]], 
                                         inv_convraicne_mat_list[sample[4]-1],
                                        mean_list[sample[4]-1], 0.33 )
            confusion_val -= decision_func(2,
                                           [sample[0], sample[1], sample[2], sample[3]],
                                           inv_convraicne_mat_list[class_idx],
                                           mean_list[class_idx], 0.33 )
            if (confusion_val > 0.0):
                confusion_matrix[sample[4]-1][class_idx] -= 1
            else:
                confusion_matrix[sample[4]-1][sample[4]-1] += 1

print("HW2: Confusion Matrix")
print_NbyN_matrix(3, confusion_matrix)
print('---------------------------------------------------')

plt.figure()

plt.title("Training/Data")
plot_2D_samples(training_data)
plt.legend(loc='upper left')
plt.show()

plt.figure()
plt.title("Samples/Decision boundaries")
plot_2D_samples(test_data)
for outter_class_idx in range(3):
    for class_idx in range(outter_class_idx + 1, 3):
        if (outter_class_idx != class_idx):
            plot_2D_boundaries([0.0, 12.0], [0.0, 7.0], 0.1,
                              2,
                              inv_convraicne_mat_list[outter_class_idx], inv_convraicne_mat_list[class_idx],
                              mean_list[outter_class_idx], mean_list[class_idx],
                              0.3333,
                              outter_class_idx,
                              class_idx)
plt.legend(loc='upper right')
plt.show()