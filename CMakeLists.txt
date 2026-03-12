cmake_minimum_required(VERSION 3.20)
project(QuantRiskEngine VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -Wpedantic")
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread)
find_package(PkgConfig REQUIRED)

# Try to find Intel MKL
find_package(MKL QUIET)
if(MKL_FOUND)
    add_definitions(-DUSE_MKL)
endif()

# Include directories
include_directories(include)
include_directories(${Boost_INCLUDE_DIRS})

# Core library
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
add_library(risk_core STATIC ${CORE_SOURCES})
target_link_libraries(risk_core ${Boost_LIBRARIES} Threads::Threads)

if(MKL_FOUND)
    target_link_libraries(risk_core ${MKL_LIBRARIES})
endif()

# Risk calculation library
file(GLOB_RECURSE RISK_SOURCES "src/risk/*.cpp")
add_library(risk_calc STATIC ${RISK_SOURCES})
target_link_libraries(risk_calc risk_core ${Boost_LIBRARIES})

# Optimization library
file(GLOB_RECURSE OPT_SOURCES "src/optimization/*.cpp")
add_library(optimization STATIC ${OPT_SOURCES})
target_link_libraries(optimization risk_core ${Boost_LIBRARIES})

# Monitoring library
file(GLOB_RECURSE MON_SOURCES "src/monitoring/*.cpp")
add_library(monitoring STATIC ${MON_SOURCES})
target_link_libraries(monitoring risk_core ${Boost_LIBRARIES})

# Main executable
add_executable(risk_engine src/main.cpp)
target_link_libraries(risk_engine
    risk_core
    risk_calc
    optimization
    monitoring
    ${Boost_LIBRARIES}
    Threads::Threads
)

# Test executable
enable_testing()
file(GLOB_RECURSE TEST_SOURCES "src/tests/*.cpp")
add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests
    risk_core
    risk_calc
    optimization
    monitoring
    ${Boost_LIBRARIES}
    Threads::Threads
)

add_test(NAME risk_engine_tests COMMAND run_tests)

# Performance benchmarks
add_executable(benchmarks src/benchmarks.cpp)
target_link_libraries(benchmarks
    risk_core
    risk_calc
    optimization
    ${Boost_LIBRARIES}
    Threads::Threads
)

# Installation
install(TARGETS risk_engine DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS risk_core risk_calc optimization monitoring DESTINATION lib)