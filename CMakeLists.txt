cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )

set(source ./liblinear/blas/blas.h ./liblinear/blas/blasp.h ./liblinear/blas/daxpy.c ./liblinear/blas/ddot.c ./liblinear/blas/dnrm2.c ./liblinear/blas/dscal.c 
./liblinear/tron.cpp ./liblinear/tron.h ./liblinear/linear.h ./liblinear/linear.cpp  
./LBF.h ./TrainModel.cpp ./TestModel.cpp ./Utils.cpp ./Tree.h ./Tree.cpp ./RandomForest.cpp ./RandomForest.h ./LBFRegressor.h ./LBFRegressor.cpp ./AlignCrop.cpp)


#if you don't want to use openMP, you should comment this line
FIND_PACKAGE( OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# add_executable(LBF ${source})
add_executable(AlignCrop ${source})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# target_link_libraries(LBF ${OpenCV_LIBS}) 
target_link_libraries(AlignCrop ${OpenCV_LIBS}) 
