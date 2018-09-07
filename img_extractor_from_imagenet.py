import glob, os, random, shutil

# dog[n02085620,n02113978]
# car[]

is_range = True

output_path = "D:\\data\\mixnet\\dog_and_car_sound_and_image\\image\\dog"
dir_from = "n02085620"
dir_to = "n02113978"

dirs = ["n02701002", "n03769881", "n03770679", "n04285008", "n04487081", "n04461696", "n04467665"]

img_cnt = 3460

img_dir = "D:\\data\\imagenet\\torrent\\ILSVRC2012_img_train"

if is_range:
    dir_names = os.listdir(img_dir)
    dir_names.sort()
    idx_from = dir_names.index(dir_from)
    idx_to = dir_names.index(dir_to)
    dirs = dir_names[idx_from:idx_to + 1]
img_cnt_per_dir = img_cnt // len(dirs) + 1

if not os.path.isdir(output_path):
    os.makedirs(output_path)

for target_dir in dirs:
    full_path = os.path.join(img_dir, target_dir)
    img_files = glob.glob(os.path.join(full_path, "*"))
    random.shuffle(img_files)
    for selected_file in img_files[:img_cnt_per_dir]:
        print(selected_file)
        if os.path.exists(os.path.join(output_path, os.path.basename(selected_file))):
            print("skip")
            continue
        shutil.copy(selected_file, os.path.join(output_path, os.path.basename(selected_file)))
