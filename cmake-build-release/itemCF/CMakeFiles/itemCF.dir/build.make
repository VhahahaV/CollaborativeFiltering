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
include itemCF/CMakeFiles/itemCF.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include itemCF/CMakeFiles/itemCF.dir/compiler_depend.make

# Include the progress variables for this target.
include itemCF/CMakeFiles/itemCF.dir/progress.make

# Include the compile flags for this target's objects.
include itemCF/CMakeFiles/itemCF.dir/flags.make

itemCF/CMakeFiles/itemCF.dir/item-cf.cpp.obj: itemCF/CMakeFiles/itemCF.dir/flags.make
itemCF/CMakeFiles/itemCF.dir/item-cf.cpp.obj: D:/C++_code/CollaborativeFiltering/itemCF/item-cf.cpp
itemCF/CMakeFiles/itemCF.dir/item-cf.cpp.obj: itemCF/CMakeFiles/itemCF.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\C++_code\CollaborativeFiltering\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object itemCF/CMakeFiles/itemCF.dir/item-cf.cpp.obj"
	cd /d D:\C++_code\CollaborativeFiltering\cmake-build-release\itemCF && C:\PROGRA~1\JETBRA~1\CLION2~1.1\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT itemCF/CMakeFiles/itemCF.dir/item-cf.cpp.obj -MF CMakeFiles\itemCF.dir\item-cf.cpp.obj.d -o CMakeFiles\itemCF.dir\item-cf.cpp.obj -c D:\C++_code\CollaborativeFiltering\itemCF\item-cf.cpp

itemCF/CMakeFiles/itemCF.dir/item-cf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/itemCF.dir/item-cf.cpp.i"
	cd /d D:\C++_code\CollaborativeFiltering\cmake-build-release\itemCF && C:\PROGRA~1\JETBRA~1\CLION2~1.1\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\C++_code\CollaborativeFiltering\itemCF\item-cf.cpp > CMakeFiles\itemCF.dir\item-cf.cpp.i

itemCF/CMakeFiles/itemCF.dir/item-cf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/itemCF.dir/item-cf.cpp.s"
	cd /d D:\C++_code\CollaborativeFiltering\cmake-build-release\itemCF && C:\PROGRA~1\JETBRA~1\CLION2~1.1\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\C++_code\CollaborativeFiltering\itemCF\item-cf.cpp -o CMakeFiles\itemCF.dir\item-cf.cpp.s

# Object files for target itemCF
itemCF_OBJECTS = \
"CMakeFiles/itemCF.dir/item-cf.cpp.obj"

# External object files for target itemCF
itemCF_EXTERNAL_OBJECTS =

itemCF/itemCF.exe: itemCF/CMakeFiles/itemCF.dir/item-cf.cpp.obj
itemCF/itemCF.exe: itemCF/CMakeFiles/itemCF.dir/build.make
itemCF/itemCF.exe: itemCF/CMakeFiles/itemCF.dir/linkLibs.rsp
itemCF/itemCF.exe: itemCF/CMakeFiles/itemCF.dir/objects1.rsp
itemCF/itemCF.exe: itemCF/CMakeFiles/itemCF.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\C++_code\CollaborativeFiltering\cmake-build-release\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable itemCF.exe"
	cd /d D:\C++_code\CollaborativeFiltering\cmake-build-release\itemCF && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\itemCF.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
itemCF/CMakeFiles/itemCF.dir/build: itemCF/itemCF.exe
.PHONY : itemCF/CMakeFiles/itemCF.dir/build

itemCF/CMakeFiles/itemCF.dir/clean:
	cd /d D:\C++_code\CollaborativeFiltering\cmake-build-release\itemCF && $(CMAKE_COMMAND) -P CMakeFiles\itemCF.dir\cmake_clean.cmake
.PHONY : itemCF/CMakeFiles/itemCF.dir/clean

itemCF/CMakeFiles/itemCF.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\C++_code\CollaborativeFiltering D:\C++_code\CollaborativeFiltering\itemCF D:\C++_code\CollaborativeFiltering\cmake-build-release D:\C++_code\CollaborativeFiltering\cmake-build-release\itemCF D:\C++_code\CollaborativeFiltering\cmake-build-release\itemCF\CMakeFiles\itemCF.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : itemCF/CMakeFiles/itemCF.dir/depend

