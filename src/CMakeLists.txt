cmake_minimum_required (VERSION 3.10)
project (NBODY)
include(common.cmake)

# ----- KOKKOS ------
add_executable(NBODY_SERIAL_LEFT main.cpp)
target_link_libraries(NBODY_SERIAL_LEFT Kokkos::kokkos)
target_compile_definitions(NBODY_SERIAL_LEFT PRIVATE ACTIVATE_KOKKOS_SERIAL_LEFT)

add_executable(NBODY_SERIAL_RIGHT main.cpp)
target_link_libraries(NBODY_SERIAL_RIGHT Kokkos::kokkos)
target_compile_definitions(NBODY_SERIAL_RIGHT PRIVATE ACTIVATE_KOKKOS_SERIAL_RIGHT)

add_executable(NBODY_OPENMP_LEFT main.cpp)
target_link_libraries(NBODY_OPENMP_LEFT Kokkos::kokkos)
target_compile_definitions(NBODY_OPENMP_LEFT PRIVATE ACTIVATE_KOKKOS_OPENMP_LEFT)

add_executable(NBODY_OPENMP_RIGHT main.cpp)
target_link_libraries(NBODY_OPENMP_RIGHT Kokkos::kokkos)
target_compile_definitions(NBODY_OPENMP_RIGHT PRIVATE ACTIVATE_KOKKOS_OPENMP_RIGHT)

add_executable(NBODY_CUDA_LEFT main.cpp)
target_link_libraries(NBODY_CUDA_LEFT Kokkos::kokkos)
target_compile_definitions(NBODY_CUDA_LEFT PRIVATE ACTIVATE_KOKKOS_CUDA_LEFT)

add_executable(NBODY_CUDA_RIGHT main.cpp)
target_link_libraries(NBODY_CUDA_RIGHT Kokkos::kokkos)
target_compile_definitions(NBODY_CUDA_RIGHT PRIVATE ACTIVATE_KOKKOS_CUDA_RIGHT)

add_executable(NBODY_SIMULATION main.cpp)
target_link_libraries(NBODY_SIMULATION Kokkos::kokkos)
target_compile_definitions(NBODY_SIMULATION PRIVATE ACTIVATE_SIMULATION)

# ----- KOKKOS WITH GRAPHS ------
add_executable(NBODY_GRAPH_CUDA_LEFT main.cpp)
target_link_libraries(NBODY_GRAPH_CUDA_LEFT Kokkos::kokkos)
target_compile_definitions(NBODY_GRAPH_CUDA_LEFT PRIVATE ACTIVATE_GRAPH_KOKKOS_CUDA_LEFT)

add_executable(NBODY_GRAPH_CUDA_RIGHT main.cpp)
target_link_libraries(NBODY_GRAPH_CUDA_RIGHT Kokkos::kokkos)
target_compile_definitions(NBODY_GRAPH_CUDA_RIGHT PRIVATE ACTIVATE_GRAPH_KOKKOS_CUDA_RIGHT)