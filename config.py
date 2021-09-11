
tiff_data_fd = r'./train_data\tiffData'
tiff_mask_fd = r'./train_data\tiff_data_mask'
terrain_png_fd = r'./train_data\png_norm'

log_dir = './logs'

train_style = [
    'srtm_11_01',  #s0
    'srtm_17_05',  #s5
    #'srtm_18_05',   #s6
    #'srtm_62_16',  #s8
    #'srtm_12_02',  #s1
    'srtm_13_05',  #s2
    'srtm_16_01',  #s3
    #'srtm_16_08',  #s4
    #'srtm_60_18',  #s7
    #'srtm_66_18',  #s9
]

typeCount = len(train_style)

scaleCount = 2

ter_size = 512


sep_conv = True

leaky_relu = True

if ter_size == 256:
    batch_size = 12
elif ter_size == 512:
    batch_size = 10

generator_train_mag = 2     # 1

accumulate_count = 6

iterations = int(120000/batch_size * typeCount / generator_train_mag)

sample_interval = int(2000/batch_size * typeCount / generator_train_mag)

save_interval = int(7500/batch_size * typeCount / generator_train_mag)

save_start = int(100/batch_size * typeCount / generator_train_mag)
