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
CMAKE_SOURCE_DIR = /tmp/ProgrammationGraphique/Part1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /tmp/ProgrammationGraphique/Part1/build

# Include any dependencies generated for this target.
include CMakeFiles/code04_dessin.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/code04_dessin.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/code04_dessin.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/code04_dessin.dir/flags.make

CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.o: CMakeFiles/code04_dessin.dir/flags.make
CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.o: ../Cours1/code04_dessin.cpp
CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.o: CMakeFiles/code04_dessin.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/ProgrammationGraphique/Part1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.o -MF CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.o.d -o CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.o -c /tmp/ProgrammationGraphique/Part1/Cours1/code04_dessin.cpp

CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/ProgrammationGraphique/Part1/Cours1/code04_dessin.cpp > CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.i

CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/ProgrammationGraphique/Part1/Cours1/code04_dessin.cpp -o CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.s

CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.o: CMakeFiles/code04_dessin.dir/flags.make
CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.o: ../Common/shaders_utilities.cpp
CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.o: CMakeFiles/code04_dessin.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/ProgrammationGraphique/Part1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.o -MF CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.o.d -o CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.o -c /tmp/ProgrammationGraphique/Part1/Common/shaders_utilities.cpp

CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/ProgrammationGraphique/Part1/Common/shaders_utilities.cpp > CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.i

CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/ProgrammationGraphique/Part1/Common/shaders_utilities.cpp -o CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.s

# Object files for target code04_dessin
code04_dessin_OBJECTS = \
"CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.o" \
"CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.o"

# External object files for target code04_dessin
code04_dessin_EXTERNAL_OBJECTS =

code04_dessin: CMakeFiles/code04_dessin.dir/Cours1/code04_dessin.cpp.o
code04_dessin: CMakeFiles/code04_dessin.dir/Common/shaders_utilities.cpp.o
code04_dessin: CMakeFiles/code04_dessin.dir/build.make
code04_dessin: /usr/lib/x86_64-linux-gnu/libOpenGL.so
code04_dessin: /usr/lib/x86_64-linux-gnu/libGLX.so
code04_dessin: /usr/lib/x86_64-linux-gnu/libGLU.so
code04_dessin: CMakeFiles/code04_dessin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tmp/ProgrammationGraphique/Part1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable code04_dessin"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/code04_dessin.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/cmake -E copy /tmp/ProgrammationGraphique/Part1/build/./code04_dessin /tmp/ProgrammationGraphique/Part1/Cours1/

# Rule to build all files generated by this target.
CMakeFiles/code04_dessin.dir/build: code04_dessin
.PHONY : CMakeFiles/code04_dessin.dir/build

CMakeFiles/code04_dessin.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/code04_dessin.dir/cmake_clean.cmake
.PHONY : CMakeFiles/code04_dessin.dir/clean

CMakeFiles/code04_dessin.dir/depend:
	cd /tmp/ProgrammationGraphique/Part1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tmp/ProgrammationGraphique/Part1 /tmp/ProgrammationGraphique/Part1 /tmp/ProgrammationGraphique/Part1/build /tmp/ProgrammationGraphique/Part1/build /tmp/ProgrammationGraphique/Part1/build/CMakeFiles/code04_dessin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/code04_dessin.dir/depend

