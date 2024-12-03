cmake_minimum_required(VERSION 3.9)
project(integracion2)
set(CMAKE_CXX_STANDARD 11)
#set(OpenCV_DIR /usr/share/OpenCV)
find_package(OpenCV REQUIRED)
add_subdirectory(model)
add_executable(integracion2 main.cpp gnuplot.h Detection.cpp Detection.h Ventana.cpp Ventana.h)
target_link_libraries(integracion2 ${OpenCV_LIBS} model)

