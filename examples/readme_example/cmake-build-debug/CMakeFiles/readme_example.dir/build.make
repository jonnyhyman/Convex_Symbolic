# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/readme_example.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/readme_example.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/readme_example.dir/flags.make

CMakeFiles/readme_example.dir/main.c.o: CMakeFiles/readme_example.dir/flags.make
CMakeFiles/readme_example.dir/main.c.o: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/readme_example.dir/main.c.o"
	/usr/bin/gcc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/readme_example.dir/main.c.o   -c /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/main.c

CMakeFiles/readme_example.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/readme_example.dir/main.c.i"
	/usr/bin/gcc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/main.c > CMakeFiles/readme_example.dir/main.c.i

CMakeFiles/readme_example.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/readme_example.dir/main.c.s"
	/usr/bin/gcc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/main.c -o CMakeFiles/readme_example.dir/main.c.s

CMakeFiles/readme_example.dir/main.c.o.requires:

.PHONY : CMakeFiles/readme_example.dir/main.c.o.requires

CMakeFiles/readme_example.dir/main.c.o.provides: CMakeFiles/readme_example.dir/main.c.o.requires
	$(MAKE) -f CMakeFiles/readme_example.dir/build.make CMakeFiles/readme_example.dir/main.c.o.provides.build
.PHONY : CMakeFiles/readme_example.dir/main.c.o.provides

CMakeFiles/readme_example.dir/main.c.o.provides.build: CMakeFiles/readme_example.dir/main.c.o


CMakeFiles/readme_example.dir/problem.c.o: CMakeFiles/readme_example.dir/flags.make
CMakeFiles/readme_example.dir/problem.c.o: ../problem.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/readme_example.dir/problem.c.o"
	/usr/bin/gcc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/readme_example.dir/problem.c.o   -c /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/problem.c

CMakeFiles/readme_example.dir/problem.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/readme_example.dir/problem.c.i"
	/usr/bin/gcc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/problem.c > CMakeFiles/readme_example.dir/problem.c.i

CMakeFiles/readme_example.dir/problem.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/readme_example.dir/problem.c.s"
	/usr/bin/gcc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/problem.c -o CMakeFiles/readme_example.dir/problem.c.s

CMakeFiles/readme_example.dir/problem.c.o.requires:

.PHONY : CMakeFiles/readme_example.dir/problem.c.o.requires

CMakeFiles/readme_example.dir/problem.c.o.provides: CMakeFiles/readme_example.dir/problem.c.o.requires
	$(MAKE) -f CMakeFiles/readme_example.dir/build.make CMakeFiles/readme_example.dir/problem.c.o.provides.build
.PHONY : CMakeFiles/readme_example.dir/problem.c.o.provides

CMakeFiles/readme_example.dir/problem.c.o.provides.build: CMakeFiles/readme_example.dir/problem.c.o


# Object files for target readme_example
readme_example_OBJECTS = \
"CMakeFiles/readme_example.dir/main.c.o" \
"CMakeFiles/readme_example.dir/problem.c.o"

# External object files for target readme_example
readme_example_EXTERNAL_OBJECTS =

readme_example: CMakeFiles/readme_example.dir/main.c.o
readme_example: CMakeFiles/readme_example.dir/problem.c.o
readme_example: CMakeFiles/readme_example.dir/build.make
readme_example: CMakeFiles/readme_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable readme_example"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/readme_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/readme_example.dir/build: readme_example

.PHONY : CMakeFiles/readme_example.dir/build

CMakeFiles/readme_example.dir/requires: CMakeFiles/readme_example.dir/main.c.o.requires
CMakeFiles/readme_example.dir/requires: CMakeFiles/readme_example.dir/problem.c.o.requires

.PHONY : CMakeFiles/readme_example.dir/requires

CMakeFiles/readme_example.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/readme_example.dir/cmake_clean.cmake
.PHONY : CMakeFiles/readme_example.dir/clean

CMakeFiles/readme_example.dir/depend:
	cd /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/cmake-build-debug /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/cmake-build-debug /mnt/c/Users/__jonny__/Dropbox/Compute/convexoptimization/convex_symbolic/examples/readme_example/cmake-build-debug/CMakeFiles/readme_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/readme_example.dir/depend

