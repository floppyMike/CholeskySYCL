cmake_minimum_required(VERSION 3.5) # setting this is required
project(SYCLCholesky)               # this sets the project name

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(AdaptiveCpp CONFIG REQUIRED)

file(GLOB SOURCES src/*.cpp)

foreach(SOURCE IN LISTS SOURCES)
	get_filename_component(EXE ${SOURCE} NAME_WE)
	add_executable(${EXE} ${SOURCE})
	if(USE_DOUBLE)
		target_compile_definitions(${EXE} PRIVATE USE_DOUBLE)
	endif()
	target_compile_options(${EXE} PUBLIC -std=c++23 -Wall -Ofast)
	target_include_directories(${EXE} PUBLIC include)
	add_sycl_to_target(TARGET ${EXE} SOURCES ${SOURCE})
endforeach()
