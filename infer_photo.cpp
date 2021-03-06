#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include "mtcnn/detector.h"
#include "draw.hpp"
#include "recognition.hpp"
#include "dataset_face/dataset_hdf5.hpp"
#include "dataset_face/dataset_proto.hpp"

#include <iostream>
#include <string>
#include <vector>

#include "H5Cpp.h"

/*
echo "" | g++ -xc - -v -E

rm -rf build;mkdir build;cd build;cmake -DCMAKE_C_COMPILER=clang \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_PREFIX_PATH="$PWD/libtorch;/usr/local/Cellar/hdf5/1.12.0;/usr/local/Cellar/protobuf/3.12.4" ..;make VERBOSE=1;cd ..


./build/infer_photo ./models ./data/got.jpg ./data/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962_traced_model.pt ./data/dataset_targarien.h5
./build/infer_photo ./models ./data/test1.jpg ./data/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962_traced_model.pt ./data/dataset_targarien.h5
./build/infer_photo ./models ./data/test4.jpg ./data/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962_traced_model.pt ./data/dataset_targarien.h5

./build/infer_photo ./models ./data/got.jpg ./data/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962_traced_model.pt ./data/dataset_targarien.protobuf
./build/infer_photo ./models ./data/test1.jpg ./data/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962_traced_model.pt ./data/dataset_targarien.protobuf
./build/infer_photo ./models ./data/test4.jpg ./data/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962_traced_model.pt ./data/dataset_targarien.protobuf

*/


int main(int argc, char **argv) {

  if (argc < 5) {
        std::cerr << "Usage " << ": "
            << "<app_binary> "
            << "<path_to_face_detection_models_dir>"
            << "<path_to_test_image>"
            << "<path_to_face_recognition_model>"
            << "<path_to_hdf5_face_database>\n";
        return 1;
    return -1;
  }
  
  std::cout.precision(17);

  std::string modelPath = argv[1];

  ProposalNetwork::Config pConfig;
  pConfig.caffeModel = modelPath + "/det1.caffemodel";
  pConfig.protoText = modelPath + "/det1.prototxt";
  pConfig.threshold = 0.6f;

  RefineNetwork::Config rConfig;
  rConfig.caffeModel = modelPath + "/det2.caffemodel";
  rConfig.protoText = modelPath + "/det2.prototxt";
  rConfig.threshold = 0.7f;

  OutputNetwork::Config oConfig;
  oConfig.caffeModel = modelPath + "/det3.caffemodel";
  oConfig.protoText = modelPath + "/det3.prototxt";
  oConfig.threshold = 0.7f;

  MTCNNDetector detector(pConfig, rConfig, oConfig);

  std::vector<Face> faces;

  std::string faceRecogintionModelPath = argv[3];
  torch::jit::script::Module module = torchInitModule(faceRecogintionModelPath);

  std::string databasePath = argv[4];
  std::vector<DatasetFace> datasetFaces;
  if(databasePath.substr(databasePath.find_last_of(".") + 1) == "h5") {
      cout << "Dataset Type is HDF5"  << endl;
      datasetFaces = readDatasetFacesFromHDF5(databasePath);
  } else if(databasePath.substr(databasePath.find_last_of(".") + 1) == "protobuf") {
      cout << "Dataset Type is PROTOBUF"  << endl;
      datasetFaces = readDatasetFacesFromProtobuf(databasePath);
  } else {
      cerr << "ERROR: Can't get database file type" << endl;
      return -1;
  }
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//--------------------------------Per Image Start------------------------------------

  cv::Mat img = cv::imread(argv[2]);

  unsigned imageWidth = img.size().width;
  unsigned imageHeight = img.size().height; 

  // Face Detection
  {
    faces = detector.detect(img, 20.f, 0.709f);
  }
  std::cout << "Number of faces found in the supplied image - " << faces.size() << std::endl;

  // Face Recognition
  for (size_t i = 0; i < faces.size(); ++i) 
  {
    cv::Mat faceImage = cropFaceImage(faces[i], img);
    faces[i].recognitionTensor = torchFaceRecognitionInference(module, faceImage);
  }
  faces = readDatasetFacesAndGetLabels(datasetFaces, faces);

  // Show Result
  auto resultImg = drawRectsAndPoints(img, faces);
  cv::imshow("test-oc", resultImg);

  cv::waitKey(0);


//-----------------------------------Per Image END-----------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------

  return 0;
}

