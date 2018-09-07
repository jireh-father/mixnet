import glob, os, random, shutil

output_path = "D:\\data\\mixnet\\dog_and_car_sound_and_image\\sound\\dog"

target_label = "3"  # 1 car horn, 3 dog bark
sound_dir = "D:\\data\\urban_sound_8k\\UrbanSound8K\\audio"

if not os.path.isdir(output_path):
    os.makedirs(output_path)

dirs = glob.glob(os.path.join(sound_dir, "fold*"))

for target_dir in dirs:
    sound_files = glob.glob(os.path.join(target_dir, "*.wav"))
    for sound_file in sound_files:
        label = os.path.basename(sound_file).split('-')[1]
        if label != target_label:
            continue
        print(sound_file)
        shutil.copy(sound_file, os.path.join(output_path, os.path.basename(sound_file)))
