#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include "mtcnn/detector.h"
#include "draw.hpp"
#include "recognition.hpp"

#include <iostream>
#include <string>
#include <vector>

#include "H5Cpp.h"
/*
echo "" | g++ -xc - -v -E

rm -rf build;mkdir build;cd build;cmake -DCMAKE_C_COMPILER=clang \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_PREFIX_PATH="$PWD/libtorch;/usr/local/Cellar/hdf5/1.12.0" ..;make VERBOSE=1;cd ..


./build/infer_cam ./models ./data/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962_traced_model.pt ./data/dataset_golovan.h5

*/
using namespace cv;
using std::cout; using std::cerr; using std::endl;


int main(int argc, char **argv)
{
/////////////////////////////////////////////start init

    if (argc < 4) 
    {
            std::cerr << "Usage " << ": "
                << "<app_binary> "
                << "<path_to_face_detection_models_dir>"
                << "<path_to_face_recognition_model>"
                << "<path_to_hdf5_face_database>\n";
            return 1;
        return -1;
    }

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


    unsigned captureWidth = 640;
    unsigned captureHeight = 480; 
    std::string windowTitle = "Recognition";


    /////////// --- Recognition Init start
    std::cout.precision(17);

    std::string faceRecogintionModelPath = argv[2];
    torch::jit::script::Module module = torchInitModule(faceRecogintionModelPath);

    std::string databasePath = argv[3];
    H5::H5File *file = readHDF5AndGroupNames(databasePath);
    
    if (!file) 
    {
        std::cout << "ERROR Getting HDF5 File" << std::endl;
        return -1;
    }
    /////////// --- Recognition Init end

/////////////////////////////////////////////end init
    Mat frame;
    cout << "Opening camera..." << endl;
    VideoCapture capture(0); // open the first camera
    if (!capture.isOpened())
    {
        cerr << "ERROR: Can't initialize camera capture" << endl;
        return 1;
    }

    capture.set(CAP_PROP_FRAME_WIDTH,captureWidth);
    capture.set(CAP_PROP_FRAME_HEIGHT,captureHeight);

    cout << "Frame width: " << capture.get(CAP_PROP_FRAME_WIDTH) << endl;
    cout << "     height: " << capture.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "Capturing FPS: " << capture.get(CAP_PROP_FPS) << endl;

    cout << endl << "Press 'ESC' to quit, 'space' to toggle frame processing" << endl;
    cout << endl << "Start grabbing..." << endl;

    size_t nFrames = 0;
    bool enableProcessing = false;
    int64 t0 = cv::getTickCount();
    int64 processingTime = 0;
    for (;;)
    {
        capture >> frame; // read the next frame from camera
        if (frame.empty())
        {
            cerr << "ERROR: Can't grab camera frame." << endl;
            break;
        }
        nFrames++;
        if (nFrames % 10 == 0)
        {
            const int N = 10;
            int64 t1 = cv::getTickCount();
            cout << "Frames captured: " << cv::format("%5lld", (long long int)nFrames)
                 << "    Average FPS: " << cv::format("%9.1f", (double)getTickFrequency() * N / (t1 - t0))
                 << "    Average time per frame: " << cv::format("%9.2f ms", (double)(t1 - t0) * 1000.0f / (N * getTickFrequency()))
                 << "    Average processing time: " << cv::format("%9.2f ms", (double)(processingTime) * 1000.0f / (N * getTickFrequency()))
                 << std::endl;
            t0 = t1;
            processingTime = 0;
        }
        if (!enableProcessing)
        {
            /////////////////////////////////////////////start face processing
            std::vector<Face> faces;

            // Face Detection
            {
                faces = detector.detect(frame, 20.f, 0.709f);
            }
            std::cout << "Number of faces found in the supplied image - " << faces.size() << std::endl;

            // Face Recognition
            for (size_t i = 0; i < faces.size(); ++i) 
            {
                cv::Mat faceImage = cropFaceImage(faces[i], frame);
                faces[i].recognitionTensor = torchFaceRecognitionInference(module, faceImage);
            }

            faces = readHDF5AndGetLabels(file, faces);
            
            // Show Result
            auto resultImg = drawRectsAndPoints(frame, faces);
            cv::imshow(windowTitle, resultImg);

            /////////////////////////////////////////////end face processing
        }
        else
        {
            int64 tp0 = cv::getTickCount();
            Mat processed;
            cv::Canny(frame, processed, 400, 1000, 5);
            processingTime += cv::getTickCount() - tp0;
            imshow("Frame", processed);
        }
        int key = waitKey(1);
        if (key == 27/*ESC*/)
            break;
        if (key == 32/*SPACE*/)
        {
            enableProcessing = !enableProcessing;
            cout << "Enable frame processing ('space' key): " << enableProcessing << endl;
        }
    }
    std::cout << "Number of captured frames: " << nFrames << endl;
    delete file;
    return nFrames > 0 ? 0 : 1;
}
