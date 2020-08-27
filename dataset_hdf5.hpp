#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include "H5Cpp.h"
std::vector<std::string> groupNames;
// Operator function
extern "C" herr_t file_info(hid_t loc_id, const char *name, const H5L_info2_t *linfo,
    void *opdata);


std::vector<DatasetFace> readDatasetFacesFromHDF5(std::string databasePath) 
{

    std::vector<DatasetFace> datasetFaces;

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


        std::cout << "Extracted group Names: \n";
        for (unsigned long i=0; i<groupNames.size();i++ )
        {
            std::cout << "\nExtracted name: " << groupNames[i] << std::endl;

            H5std_string groupName( groupNames[i] );

            H5::Group* group = new H5::Group(file->openGroup(groupName));

            H5::DataSet* dataset;
            try {  // to determine if the dataset exists in the group
                 dataset = new H5::DataSet( group->openDataSet( "embedding" ));
            }
            catch( H5::GroupIException not_found_error ) {
                std::cout << "\t ERROR: Dataset is not found." << std::endl;
                // return 0;
                return datasetFaces;
            }
            
            // Read Embed
            double  embedding[512];
            dataset->read(embedding, H5::PredType::NATIVE_DOUBLE);
            // std::cout << "\n\n\tArray Extracted embedding examples: " << std::endl;

            // Array to Tensor conversion V1 (shares memory)
            // auto options = torch::TensorOptions().dtype(torch::kFloat64);
            // at::Tensor embTensor = torch::from_blob(embedding, {1, 512}, options); 
            // std::cout << "\tTensor Output slice 0-5: " << embTensor.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;

            // Array to Tensor conversion V2 (copy memory)
            auto embTensor = torch::zeros( {1, 512},torch::kF64);
            std::memcpy(embTensor.data_ptr(),embedding,sizeof(double)*embTensor.numel());
            std::cout << "\tTensor Output slice 0-5: " << embTensor.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;


            // Adding to dataset;
            DatasetFace df = DatasetFace(groupNames[i], embTensor);

            std::cout << "+++NAME: " << df.getName() << "\n";
            std::cout << "+++EMB: " << df.getEmbeddingTensor().slice(/*dim=*/1, /*start=*/0, /*end=*/5) << "\n";
            std::cout << "+++OBJ ADDR: " << &df << std::endl;

            datasetFaces.push_back(df);

            std::cout << "\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n";
            for (unsigned long z=0; z<datasetFaces.size();z++ )
            {   
                at::Tensor tsr = datasetFaces[z].getEmbeddingTensor(); 
                std::cout << "NAME: " << datasetFaces[z].getName() << std::endl;
                std::cout << "EMB: " << tsr.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
                
                std::cout << "OBJ ADDR: " << &(datasetFaces[z]) << std::endl;
                std::cout << "EMB: " << &tsr << std::endl;

            }
            
            std::cout << "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n";

        }

        return datasetFaces;
    }  // end of try block

    // catch failure caused by the H5File operations
    catch( H5::FileIException error )
    {
        error.printErrorStack();
        return datasetFaces;
    }

    // catch failure caused by the DataSet operations
    catch( H5::DataSetIException error )
    {
        error.printErrorStack();
        return datasetFaces;
    }

    // catch failure caused by the DataSpace operations
    catch( H5::DataSpaceIException error )
    {
        error.printErrorStack();
        return datasetFaces;
    }

    // catch failure caused by the Attribute operations
    catch( H5::AttributeIException error )
    {
        error.printErrorStack();
        return datasetFaces;
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
    // std::cout << "Name : " << name << std::endl;

    H5Gclose(group);
    return 0;
}
