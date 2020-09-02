#ifndef _include_dataset_face_proto_h_
#define _include_dataset_face_proto_h_

#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include "DatasetFace.cpp"

#include <iostream>
#include <fstream>
#include <string>
#include "face_dataset.pb.h"
using namespace std;

std::vector<DatasetFace> readDatasetFacesFromProtobuf(std::string databasePath) 
{

    std::vector<DatasetFace> datasetFaces;

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    dataset_faces::DatasetObject dataset_object;
    {
        // Read the existing face dataset .
        fstream input(databasePath, ios::in | ios::binary);
        if (!dataset_object.ParseFromIstream(&input)) {
            cerr << "Failed to parse face dataset object." << endl;
            return datasetFaces;
        }
    }



    for (int i = 0; i < dataset_object.faceobjects_size(); i++) {
        const dataset_faces::FaceObject& faceObject = dataset_object.faceobjects(i);

        cout << "\n  Name: " << faceObject.name() << endl;


        double embedding[512];
        // getting data from protobuf
        for (int j = 0; j < faceObject.embeddings_size(); j++) 
        {
            embedding[j] = faceObject.embeddings(j);
        }

        // Array to Tensor conversion V2 (copy memory)
        auto embTensor = torch::zeros( {1, 512},torch::kF64);
        std::memcpy(embTensor.data_ptr(),embedding,sizeof(double)*embTensor.numel());
        std::cout << "\tTensor Output slice 0-5: " << embTensor.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;

        // Adding to dataset;
        DatasetFace df = DatasetFace(faceObject.name(), embTensor);

        std::cout << "+++NAME: " << df.getName() << "\n";
        std::cout << "+++EMB: " << df.getEmbeddingTensor().slice(/*dim=*/1, /*start=*/0, /*end=*/5) << "\n";
        std::cout << "+++OBJ ADDR: " << &df << std::endl;

        datasetFaces.push_back(df);


        // // printing embedding
        // for (int j = 0; j < (sizeof(embedding)/sizeof(*embedding)); j++) 
        // cout << embedding[j];

    }

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();


    return datasetFaces;

}


#endif
