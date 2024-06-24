cmake_minimum_required(VERSION 3.15)
project(pcl_project)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set vcpkg toolchain file
if(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "Please set VCPKG_ROOT to the path of your vcpkg installation")
endif()

set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Specify vcpkg installed directory
set(CMAKE_PREFIX_PATH "$ENV{VCPKG_ROOT}/installed/x64-linux")

# Find PCL package
find_package(PCL REQUIRED COMPONENTS common io visualization)

# Find CGAL package
find_package(CGAL REQUIRED)

# Include directories
include_directories(${PCL_INCLUDE_DIRS} ${CGAL_INCLUDE_DIRS})

# Link directories
link_directories(${PCL_LIBRARY_DIRS} ${CGAL_LIBRARY_DIRS})

# Add executable
add_executable(pcl_project main.cpp)

# Link libraries
target_link_libraries(pcl_project ${PCL_LIBRARIES} ${CGAL_LIBRARIES})

# Add definitions
add_definitions(${PCL_DEFINITIONS})
