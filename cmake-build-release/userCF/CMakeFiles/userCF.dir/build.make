# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.26

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2023.2.1\bin\cmake\win\x64\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2023.2.1\bin\cmake\win\x64\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\C++_code\CollaborativeFiltering

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\C++_code\CollaborativeFiltering\cmake-build-release

# Include any dependencies generated for this target.
include userCF/CMakeFiles/userCF.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include userCF/CMakeFiles/userCF.dir/compiler_depend.make

# Include the progress variables for this target.
include userCF/CMakeFiles/userCF.dir/progress.make

# Include the compile flags for this target's objects.
include userCF/CMakeFiles/userCF.dir/flags.make

userCF/CMakeFiles/userCF.dir/user-cf.cpp.obj: userCF/CMakeFiles/userCF.dir/flags.make
userCF/CMakeFiles/userCF.dir/user-cf.cpp.obj: D:/C++_code/CollaborativeFiltering/userCF/user-cf.cpp
userCF/CMakeFiles/userCF.dir/user-cf.cpp.obj: userCF/CMakeFiles/userCF.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\C++_code\CollaborativeFiltering\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object userCF/CMakeFiles/userCF.dir/user-cf.cpp.obj"
	cd /d D:\C++_code\CollaborativeFiltering\cmake-build-release\userCF && C:\PROGRA~1\JETBRA~1\CLION2~1.1\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT userCF/CMakeFiles/userCF.dir/user-cf.cpp.obj -MF CMakeFiles\userCF.dir\user-cf.cpp.obj.d -o CMakeFiles\userCF.dir\user-cf.cpp.obj -c D:\C++_code\CollaborativeFiltering\userCF\user-cf.cpp

userCF/CMakeFiles/userCF.dir/user-cf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/userCF.dir/user-cf.cpp.i"
	cd /d D:\C++_code\CollaborativeFiltering\cmake-build-release\userCF && C:\PROGRA~1\JETBRA~1\CLION2~1.1\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\C++_code\CollaborativeFiltering\userCF\user-cf.cpp > CMakeFiles\userCF.dir\user-cf.cpp.i

userCF/CMakeFiles/userCF.dir/user-cf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/userCF.dir/user-cf.cpp.s"
	cd /d D:\C++_code\CollaborativeFiltering\cmake-build-release\userCF && C:\PROGRA~1\JETBRA~1\CLION2~1.1\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\C++_code\CollaborativeFiltering\userCF\user-cf.cpp -o CMakeFiles\userCF.dir\user-cf.cpp.s

# Object files for target userCF
userCF_OBJECTS = \
"CMakeFiles/userCF.dir/user-cf.cpp.obj"

# External object files for target userCF
userCF_EXTERNAL_OBJECTS =

userCF/userCF.exe: userCF/CMakeFiles/userCF.dir/user-cf.cpp.obj
userCF/userCF.exe: userCF/CMakeFiles/userCF.dir/build.make
userCF/userCF.exe: userCF/CMakeFiles/userCF.dir/linkLibs.rsp
userCF/userCF.exe: userCF/CMakeFiles/userCF.dir/objects1.rsp
userCF/userCF.exe: userCF/CMakeFiles/userCF.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\C++_code\CollaborativeFiltering\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable userCF.exe"
	cd /d D:\C++_code\CollaborativeFiltering\cmake-build-release\userCF && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\userCF.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
userCF/CMakeFiles/userCF.dir/build: userCF/userCF.exe
.PHONY : userCF/CMakeFiles/userCF.dir/build

userCF/CMakeFiles/userCF.dir/clean:
	cd /d D:\C++_code\CollaborativeFiltering\cmake-build-release\userCF && $(CMAKE_COMMAND) -P CMakeFiles\userCF.dir\cmake_clean.cmake
.PHONY : userCF/CMakeFiles/userCF.dir/clean

userCF/CMakeFiles/userCF.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\C++_code\CollaborativeFiltering D:\C++_code\CollaborativeFiltering\userCF D:\C++_code\CollaborativeFiltering\cmake-build-release D:\C++_code\CollaborativeFiltering\cmake-build-release\userCF D:\C++_code\CollaborativeFiltering\cmake-build-release\userCF\CMakeFiles\userCF.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : userCF/CMakeFiles/userCF.dir/depend
