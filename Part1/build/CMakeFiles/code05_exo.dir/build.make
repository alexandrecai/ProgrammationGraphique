# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_SOURCE_DIR = /tmp/programmationgraphique/Part1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /tmp/programmationgraphique/Part1/build

# Include any dependencies generated for this target.
include CMakeFiles/code05_exo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/code05_exo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/code05_exo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/code05_exo.dir/flags.make

CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.o: CMakeFiles/code05_exo.dir/flags.make
CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.o: ../Cours1/code05_exo.cpp
CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.o: CMakeFiles/code05_exo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/programmationgraphique/Part1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.o -MF CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.o.d -o CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.o -c /tmp/programmationgraphique/Part1/Cours1/code05_exo.cpp

CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/programmationgraphique/Part1/Cours1/code05_exo.cpp > CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.i

CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/programmationgraphique/Part1/Cours1/code05_exo.cpp -o CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.s

CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.o: CMakeFiles/code05_exo.dir/flags.make
CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.o: ../Common/shaders_utilities.cpp
CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.o: CMakeFiles/code05_exo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/programmationgraphique/Part1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.o -MF CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.o.d -o CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.o -c /tmp/programmationgraphique/Part1/Common/shaders_utilities.cpp

CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/programmationgraphique/Part1/Common/shaders_utilities.cpp > CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.i

CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/programmationgraphique/Part1/Common/shaders_utilities.cpp -o CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.s

# Object files for target code05_exo
code05_exo_OBJECTS = \
"CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.o" \
"CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.o"

# External object files for target code05_exo
code05_exo_EXTERNAL_OBJECTS =

code05_exo: CMakeFiles/code05_exo.dir/Cours1/code05_exo.cpp.o
code05_exo: CMakeFiles/code05_exo.dir/Common/shaders_utilities.cpp.o
code05_exo: CMakeFiles/code05_exo.dir/build.make
code05_exo: /usr/lib/x86_64-linux-gnu/libOpenGL.so
code05_exo: /usr/lib/x86_64-linux-gnu/libGLX.so
code05_exo: /usr/lib/x86_64-linux-gnu/libGLU.so
code05_exo: CMakeFiles/code05_exo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tmp/programmationgraphique/Part1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable code05_exo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/code05_exo.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/cmake -E copy /tmp/programmationgraphique/Part1/build/./code05_exo /tmp/programmationgraphique/Part1/Cours1/

# Rule to build all files generated by this target.
CMakeFiles/code05_exo.dir/build: code05_exo
.PHONY : CMakeFiles/code05_exo.dir/build

CMakeFiles/code05_exo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/code05_exo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/code05_exo.dir/clean

CMakeFiles/code05_exo.dir/depend:
	cd /tmp/programmationgraphique/Part1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tmp/programmationgraphique/Part1 /tmp/programmationgraphique/Part1 /tmp/programmationgraphique/Part1/build /tmp/programmationgraphique/Part1/build /tmp/programmationgraphique/Part1/build/CMakeFiles/code05_exo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/code05_exo.dir/depend

