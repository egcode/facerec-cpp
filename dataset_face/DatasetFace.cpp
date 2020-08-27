#ifndef _include_dataset_face_h_
#define _include_dataset_face_h_

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.


class DatasetFace {
      std::string name;
      at::Tensor embeddingTensor;
    public:
      DatasetFace(std::string n, at::Tensor et)
      {
        name = n;
        embeddingTensor = et;
      }
      std::string getName() 
      {
        return name;
      }
      at::Tensor getEmbeddingTensor() 
      {
        return embeddingTensor;
      }
};

#endif
