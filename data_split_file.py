import os
import shutil
import math

original_dataset_dir = './dataset'
classes_list = os.listdir(original_dataset_dir)

base_dir = './splitted'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'val')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)

for clss in classes_list:
    os.mkdir(os.path.join(train_dir, clss))
    os.mkdir(os.path.join(validation_dir,clss))
    os.mkdir(os.path.join(test_dir, clss))


for clss in classes_list:
    path = os.path.join(original_dataset_dir, clss)
    fnamesa = os.listdir(path) # path 위치에 있는 이미지 파일의 목록저장
    train_size = math.floor(len(fnamesa)*0.6) #비율 정하기 floor 실수를 내림 하여 저장함.
    validation_size = math.floor(len(fnamesa)*0.2)
    test_size = math.floor(len(fnamesa)*0.2)

    train_fnames = fnamesa[:train_size]
    print('Train Size(-----',clss,'-----):', len(train_fnames))
    for fnames in train_fnames:
        src = os.path.join(path, fnames)
        dst = os.path.join(os.path.join(train_dir,clss),fnames)
        shutil.copyfile(src, dst)

    validation_fnames = fnamesa[train_size:(validation_size + train_size)]
    print('Validation Size(-----', clss, '------):', len(validation_fnames))
    for fnames in validation_fnames:
        src = os.path.join(path, fnames)
        dst = os.path.join(os.path.join(validation_dir, clss), fnames)
        shutil.copyfile(src, dst)

    test_fnames = fnamesa[(train_size+validation_size):(train_size+validation_size+test_size)]
    print('Test Size(', clss, '):', len(test_fnames))
    for fnames in test_fnames:
        src = os.path.join(path, fnames)
        dst = os.path.join(os.path.join(test_dir, clss), fnames)
        shutil.copyfile(src, dst)