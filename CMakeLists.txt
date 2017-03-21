cmake_minimum_required(VERSION 3.1)


#
# Project Properties
#
PROJECT (SIR_VexCL)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FIND_PACKAGE(Boost COMPONENTS system REQUIRED)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES(.)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

set(project_SRCS SIR_VexCL.cpp)


add_executable(SIR_VexCL ${project_SRCS})
target_link_libraries(SIR_VexCL  ${Boost_SYSTEM_LIBRARY} ${OpenCL_LIBRARY})