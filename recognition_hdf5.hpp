#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include "H5Cpp.h"
std::vector<std::string> groupNames;
// Operator function
extern "C" herr_t file_info(hid_t loc_id, const char *name, const H5L_info2_t *linfo,
    void *opdata);


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

            at::Tensor empTensor = emptyTensor();

            for (size_t j = 0; j < faces.size(); ++j) 
            {
              // Distance
              // std::cout << "\n\nDistance Start------------------------------------------------: " << std::endl;

              // Check if tensor is not empty
              if (torch::equal(empTensor, faces[j].recognitionTensor) == 0)
              {
                double dist = distanceCosine(faces[j].recognitionTensor, embTensor);
                // std::cout << "-----Name : " << faces[j].label << " Distance : " << faces[j].dist << '\n'; // 1.0

                if (faces[j].dist > dist)
                {
                    faces[j].dist = dist;
                    faces[j].label = groupNames[i];
                    // std::cout << "Name : " << faces[j].label << " Distance : " << faces[j].dist << '\n'; // 1.0

                }
                // std::cout << "\n\nDistance End--------------------------------------------------: " << std::endl;

              } else {
                std::cout << "\n\n\tArray Empty Tensor, ignoring..." << std::endl;
              }

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
