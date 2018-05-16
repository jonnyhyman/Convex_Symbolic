cmake_minimum_required(VERSION 3.5)
project(readme_example C)

set(CMAKE_C_STANDARD 99)

# Build ECOS shared library
execute_process(COMMAND make shared WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ecos")

include_directories(ecos/include)
include_directories(ecos/external/SuiteSparse_config)

link_directories(ecos)
link_libraries(ecos)

add_executable(readme_example main.c problem.h problem.c)
target_link_libraries(readme_example m)  # for math.h