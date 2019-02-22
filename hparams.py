# Preprocess
cleaners = 'english_cleaners'

# Audio:
num_mels = 80
n_mels = 80
num_freq = 1025
sample_rate = 20000
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5

n_fft = 2048
n_iter = 50
# max_db = 100
# ref_db = 20

# Model:
E = 256
r = 5
hidden_size = 128
embedding_size = 256
teacher_forcing_ratio = 1.0
max_iters = 200
max_Ty = 200
# reference encoder
ref_enc_filters = [32, 32, 64, 64, 128, 128]
ref_enc_size = [3, 3]
ref_enc_strides = [2, 2]
ref_enc_pad = [1, 1]
ref_enc_gru_size = E // 2
# style token layer
token_num = 10
num_heads = 8
K = 16
decoder_K = 8
embedded_size = E
dropout_p = 0.5
num_banks = 15
num_highways = 4
vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"

# Training:
outputs_per_step = 5
batch_size = 32
epochs = 10
lr = 0.0001
clip_value = 1.
loss_weight = 0.5
# decay_step = [500000, 1000000, 2000000]
decay_step = [20, 60]
# save_step = 2000
save_step = 20
# log_step = 200
log_step = 5
clear_Time = 20
checkpoint_path = './model_new'
