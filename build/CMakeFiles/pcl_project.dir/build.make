# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/as/Projects/Log

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/as/Projects/Log/build

# Include any dependencies generated for this target.
include CMakeFiles/pcl_project.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pcl_project.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pcl_project.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pcl_project.dir/flags.make

CMakeFiles/pcl_project.dir/main.cpp.o: CMakeFiles/pcl_project.dir/flags.make
CMakeFiles/pcl_project.dir/main.cpp.o: /home/as/Projects/Log/main.cpp
CMakeFiles/pcl_project.dir/main.cpp.o: CMakeFiles/pcl_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/as/Projects/Log/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pcl_project.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pcl_project.dir/main.cpp.o -MF CMakeFiles/pcl_project.dir/main.cpp.o.d -o CMakeFiles/pcl_project.dir/main.cpp.o -c /home/as/Projects/Log/main.cpp

CMakeFiles/pcl_project.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcl_project.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/as/Projects/Log/main.cpp > CMakeFiles/pcl_project.dir/main.cpp.i

CMakeFiles/pcl_project.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcl_project.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/as/Projects/Log/main.cpp -o CMakeFiles/pcl_project.dir/main.cpp.s

# Object files for target pcl_project
pcl_project_OBJECTS = \
"CMakeFiles/pcl_project.dir/main.cpp.o"

# External object files for target pcl_project
pcl_project_EXTERNAL_OBJECTS =

pcl_project: CMakeFiles/pcl_project.dir/main.cpp.o
pcl_project: CMakeFiles/pcl_project.dir/build.make
pcl_project: vcpkg_installed/x64-linux/debug/lib/libpcl_common.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libpcl_octree.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libpcl_io_ply.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libpcl_io.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_system.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_iostreams.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_filesystem.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_serialization.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libpcl_octree.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libpcl_io_ply.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libpcl_common.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_iostreams.a
pcl_project: vcpkg_installed/x64-linux/lib/libz.a
pcl_project: vcpkg_installed/x64-linux/lib/libbz2.a
pcl_project: vcpkg_installed/x64-linux/lib/liblzma.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libzstd.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_random.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_filesystem.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_serialization.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_thread.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_atomic.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_chrono.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_system.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_date_time.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_regex.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_container.a
pcl_project: vcpkg_installed/x64-linux/debug/lib/libboost_exception.a
pcl_project: vcpkg_installed/x64-linux/lib/libpng16.a
pcl_project: vcpkg_installed/x64-linux/lib/libz.a
pcl_project: /usr/lib/gcc/x86_64-linux-gnu/12/libgomp.so
pcl_project: /usr/lib/x86_64-linux-gnu/libpthread.a
pcl_project: CMakeFiles/pcl_project.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/as/Projects/Log/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pcl_project"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pcl_project.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pcl_project.dir/build: pcl_project
.PHONY : CMakeFiles/pcl_project.dir/build

CMakeFiles/pcl_project.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pcl_project.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pcl_project.dir/clean

CMakeFiles/pcl_project.dir/depend:
	cd /home/as/Projects/Log/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/as/Projects/Log /home/as/Projects/Log /home/as/Projects/Log/build /home/as/Projects/Log/build /home/as/Projects/Log/build/CMakeFiles/pcl_project.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pcl_project.dir/depend

