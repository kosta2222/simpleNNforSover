# nn_constants.[py]
# Параметры статических массивов,количества слоев,количества эпох
max_in_nn = 4
max_trainSet_rows = 10
max_validSet_rows = 10
max_rows_orOut = 4
max_am_layer = 7
max_am_epoch = 25
max_am_objMse = max_am_epoch
max_stack_matrEl = 256
max_stack_otherOp = 4
bc_bufLen = 256 * 2
# команды для operations
RELU = 1
RELU_DERIV = 2
SIGMOID = 3
SIGMOID_DERIV = 4
TRESHOLD_FUNC = 5
TRESHOLD_FUNC_DERIV = 6
LEAKY_RELU = 7
LEAKY_RELU_DERIV = 8
INIT_W_HE = 9
INIT_W_GLOROT = 10
DEBUG = 11
DEBUG_STR = 12
