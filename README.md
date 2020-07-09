# Facerec-cpp

C++ inference implementation of [this repo](https://github.com/egcode/facerec) 



## Build 
To build binaries:
```bash
git clone https://github.com/egcode/facerec-cpp.git
cd facerec-cpp
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=clang \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_PREFIX_PATH="$PWD/libtorch;/usr/local/Cellar/hdf5/1.12.0" ..
make VERBOSE=1
cd ..
```

## Run
```bash
cd facerec-cpp

# Camera Inference example (<app_binary> <path_to_facerecognition_model_dir> <path_to_hdf5_dataset>):
./build/infer_cam ./models ./data/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962_traced_model.pt ./data/dataset_targarien.h5


# Photo Inference examples (<app_binary> <path_to_mtcnn_models_dir> <path_to_test_image> <path_to_facerecognition_model_dir> <path_to_hdf5_dataset>):

./build/infer_photo ./models ./data/got.jpg ./data/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962_traced_model.pt ./data/dataset_targarien.h5
./build/infer_photo ./models ./data/test1.jpg ./data/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962_traced_model.pt ./data/dataset_targarien.h5
./build/infer_photo ./models ./data/test4.jpg ./data/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962_traced_model.pt ./data/dataset_targarien.h5


```



### Requirements

* OpenCV 4.1+
* HDF5 1.12.0
* CMake 3.2+
