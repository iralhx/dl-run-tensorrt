#pragma once
#include <cstdlib>
#include <iostream>  
#include<cudnn.h>

#define CHECK_CUDNN(expression)                                  \
      {                                                          \
        cudnnStatus_t status = (expression);                     \
        if (status != CUDNN_STATUS_SUCCESS) {                    \
          std::cerr << "Error on line " << __LINE__ << ": "      \
                    << cudnnGetErrorString(status) << std::endl; \
          std::exit(EXIT_FAILURE);                               \
        }                                                        \
      }