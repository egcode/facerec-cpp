#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include "mtcnn/detector.h"

#include <string>
#include <vector>
#include <iostream>

#include "H5Cpp.h"
std::vector<std::string> groupNames;
// Operator function
extern "C" herr_t file_info(hid_t loc_id, const char *name, const H5L_info2_t *linfo,
    void *opdata);

double distanceCosine(at::Tensor tensor1, at::Tensor tensor2)
{
    /*
        ###### Based on `scipy`
        uv = np.average(embeddings1 * embeddings2)
        uu = np.average(np.square(embeddings1))
        vv = np.average(np.square(embeddings2))
        dist = 1.0 - uv / np.sqrt(uu * vv)
    */
    at::Tensor mult = tensor1 * tensor2;
    at::Tensor uv = mult.mean();

    at::Tensor arr1TensorSquare = tensor1.square();
    at::Tensor uu = arr1TensorSquare.mean();

    at::Tensor arr2TensorSquare = tensor2.square();
    at::Tensor vv = arr2TensorSquare.mean();

    at::Tensor uuMultVv = uu * vv;
    at::Tensor uuvvSqrt = uuMultVv.sqrt();

    at::Tensor distTensor = 1.0 - uv / uuvvSqrt;

    // convert tensor to double
    double* floatBuffer = distTensor.data_ptr<double>();
    double dist = floatBuffer[0];
    return dist;
}

cv::Mat cropFaceImage(Face face, cv::Mat img) 
{
    cv::Rect rectSquare = face.bbox.getSquare().getRect(); // Make Square
    // std::cout << "rect square: " << rectSquare << std::endl;
    // std::cout << "x: " << rectSquare.x << std::endl;
    // std::cout << "y: " << rectSquare.y << std::endl;
    // std::cout << "width: " << rectSquare.width << std::endl;
    // std::cout << "height: " << rectSquare.height << std::endl;

    // auto rect = faces[i].bbox.getRect(); 
    // std::cout << "rect: " << rect << std::endl;

  
    // Crop the full image to that image contained by the rectangle myROI
    // Note that this doesn't copy the data
    cv::Mat croppedImage = img(rectSquare);

    cv::Mat reshapedImage;
    cv::Size size(112, 112);
    cv::resize(croppedImage,reshapedImage,size);//resize(src,dst,size)

    cv::Mat resultImage;
    reshapedImage.convertTo(resultImage, CV_32FC3, 1.0f / 255.0f);
    // std::cout << "resultImage: " << resultImage.size() << '\n';
    return resultImage;
} 

torch::jit::script::Module torchInitModule(std::string modulePath)
{

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(modulePath);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading face recognition model\n";
    // return -1;
      throw "error loading face recognition model\n";
  }
  std::cout << "face recognition model OK\n";
  return module;
}

at::Tensor torchFaceRecognitionInference(torch::jit::script::Module module, cv::Mat faceImage)
{
  // Get Tensor from Mat and shuffle channels
  at::Tensor tensorFromMat = torch::from_blob(faceImage.data, {1, 112, 112, 3}); 
  torch::jit::IValue input_tensor = tensorFromMat.permute({0, 3, 1, 2});

  // Add Tensor to actual inputs
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_tensor);

  // --- Model Inference
  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  //   std::cout << "Output slice 0-5: " << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    // std::cout << "Output size: " << output.sizes() << '\n';
  return output;
}


/*
 * HDF5 Read and Get Group Names.
 */
H5::H5File * readHDF5AndGroupNames(std::string databasePath)
{
   // Try block to detect exceptions raised by any of the calls inside it
    try
    {

        H5std_string FILE_NAME( databasePath );

        /*
         * Turn off the auto-printing when failure occurs so that we can
         * handle the errors appropriately
         */
        H5::Exception::dontPrint();

        /*
         * Create the named file, truncating the existing one if any,
         * using default create and access property lists.
         */
        H5::H5File *file = new H5::H5File( FILE_NAME, H5F_ACC_RDONLY );


        /*
         * Use iterator to see the names of the objects in the file
         * root directory.
         */
        std::cout << std::endl << "Iterating over elements in the file" << std::endl;
        herr_t idx = H5Literate2(file->getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, file_info, NULL);
        std::cout << "idx: " << idx << std::endl;

        return file;
    }  // end of try block

    // catch failure caused by the H5File operations
    catch( H5::FileIException error )
    {
        error.printErrorStack();
        return NULL;
    }

    // catch failure caused by the DataSet operations
    catch( H5::DataSetIException error )
    {
        error.printErrorStack();
        return NULL;
    }

    // catch failure caused by the DataSpace operations
    catch( H5::DataSpaceIException error )
    {
        error.printErrorStack();
        return NULL;
    }

    // catch failure caused by the Attribute operations
    catch( H5::AttributeIException error )
    {
        error.printErrorStack();
        return NULL;
    }
  
}

/*
 * HDF5 Read and Get Labels.
 */
std::vector<Face> readHDF5AndGetLabels(H5::H5File *file, std::vector<Face> faces)
{
   // Try block to detect exceptions raised by any of the calls inside it
    try
    {
        std::cout << "Extracted group Names: \n";
        for (unsigned long i=0; i<groupNames.size();i++ )
        {
            // std::cout << "\n\n  extracted name: " << groupNames[i] << std::endl;

            H5std_string groupName( groupNames[i] );

            H5::Group* group = new H5::Group(file->openGroup(groupName));

            H5::DataSet* dataset;
            try {  // to determine if the dataset exists in the group
                 dataset = new H5::DataSet( group->openDataSet( "embedding" ));
            }
            catch( H5::GroupIException not_found_error ) {
                std::cout << "\t ERROR: Dataset is not found." << std::endl;
                // return 0;
                return faces;
            }
            
            // Read Embed
            double  embedding[512];
            dataset->read(embedding, H5::PredType::NATIVE_DOUBLE);
            // std::cout << "\n\n\tArray Extracted embedding examples: " << std::endl;

            // Array to Tensor conversion V1
            auto options = torch::TensorOptions().dtype(torch::kFloat64);
            at::Tensor embTensor = torch::from_blob(embedding, {1, 512}, options); 
            // std::cout << "\n\n\tTensor Output slice 0-5: " << std::endl;

            // Array to Tensor conversion V2
            // auto embTensor = torch::zeros( {1, 512},torch::kF64);
            // std::memcpy(embTensor.data_ptr(),embedding,sizeof(double)*embTensor.numel());

            for (size_t j = 0; j < faces.size(); ++j) 
            {
              // Distance
              // std::cout << "\n\nDistance Start------------------------------------------------: " << std::endl;

              double dist = distanceCosine(faces[j].recognitionTensor, embTensor);
              // std::cout << "-----Name : " << faces[j].label << " Distance : " << faces[j].dist << '\n'; // 1.0

              if (faces[j].dist > dist)
              {
                faces[j].dist = dist;
                faces[j].label = groupNames[i];
                // std::cout << "Name : " << faces[j].label << " Distance : " << faces[j].dist << '\n'; // 1.0

              }
              // std::cout << "\n\nDistance End--------------------------------------------------: " << std::endl;
            }
            

        }

        return faces;
    }  // end of try block

    // catch failure caused by the H5File operations
    catch( H5::FileIException error )
    {
        error.printErrorStack();
        // return -1;
        return faces;
    }

    // catch failure caused by the DataSet operations
    catch( H5::DataSetIException error )
    {
        error.printErrorStack();
        // return -1;
        return faces;
    }

    // catch failure caused by the DataSpace operations
    catch( H5::DataSpaceIException error )
    {
        error.printErrorStack();
        // return -1;
        return faces;
    }

    // catch failure caused by the Attribute operations
    catch( H5::AttributeIException error )
    {
        error.printErrorStack();
        // return -1;
        return faces;
    }
  
}

/*
 * HDF5 Operator function.
 */
herr_t
file_info(hid_t loc_id, const char *name, const H5L_info2_t *linfo, void *opdata)
{
    hid_t group;

    /*
     * Open the group using its name.
     */
    group = H5Gopen2(loc_id, name, H5P_DEFAULT);

    groupNames.push_back(name);

    /*
     * Display group name.
     */
    std::cout << "Name : " << name << std::endl;

    H5Gclose(group);
    return 0;
}
