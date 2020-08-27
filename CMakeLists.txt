cmake_minimum_required(VERSION 3.2)

project (MTCNN_APP)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED )
find_package(Torch REQUIRED)

find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
include_directories( ${HDF5_INCLUDE_DIR})
link_directories( ${HDF5_LIBRARY_DIRS} )

INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})

set(SOURCES
./mtcnn/detector.cpp
./mtcnn/face.h 
./mtcnn/onet.cpp 
./mtcnn/pnet.cpp ./mtcnn/rnet.cpp
./mtcnn/detector.h ./mtcnn/helpers.h ./mtcnn/onet.h ./mtcnn/pnet.h ./mtcnn/rnet.h
./draw.hpp
./recognition.hpp
./dataset_face/dataset_proto.hpp
./dataset_face/dataset_hdf5.hpp
./dataset_face/face_dataset.pb.h
./dataset_face/face_dataset.pb.cc

)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(APP_SOURCES
infer_photo.cpp
infer_cam.cpp
)

foreach( sourcefile ${APP_SOURCES} )
    # Simple string replace, to cut off .cpp.
    string( REPLACE ".cpp" "" name ${sourcefile} )
    add_executable( ${name} ${sourcefile} ${SOURCES} )
    target_link_libraries (${name} LINK_PUBLIC ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${PROTOBUF_LIBRARY})
    #set_property(TARGET ${name} PROPERTY CXX_STANDARD 14)
endforeach( sourcefile ${APP_SOURCES} )

