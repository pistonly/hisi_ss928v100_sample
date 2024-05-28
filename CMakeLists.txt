cmake_minimum_required(VERSION 3.5.1)

# project information
project(VI_AND_INFERENCE)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
add_definitions(-DSENSOR0_TYPE=OV_OS08A20_MIPI_8M_30FPS_12BIT)

# Compile options
add_compile_options(
        -fPIC
        -fstack-protector-all
        -D_FORTIFY_SOURCE=2 -O2
        -lstdc++
)

# SDK path
# set(SDK_PATH /home/liuyang/Documents/haisi/ss928_sdk_022)
set(SDK_PATH /home/liuyang/Documents/haisi/ss928_sdk_021)
set(MPP_PATH ${SDK_PATH}/smp/a55_linux/mpp)
set(SAMPLE_PATH ${MPP_PATH}/sample)
set(OPENCV /home/liuyang/Documents/haisi/opencv)


# Specify target generation path
set(CMAKE_C_COMPILER "aarch64-mix210-linux-gcc")
set(CMAKE_CXX_COMPILER "aarch64-mix210-linux-g++")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie")
set(CMAKE_SKIP_RPATH TRUE)

# Header path
include_directories(
  ./include/svp/common/
  ./include/svp/ive/
  ${MPP_PATH}/out/include/
  ${MPP_PATH}/out/include/svp_npu/
  ${SAMPLE_PATH}/common/
  ${SAMPLE_PATH}/audio/adp/
)

link_directories(
  ${MPP_PATH}/out/lib
  ${MPP_PATH}/out/lib/svp_npu
  ${OPENCV}/lib
)

# FILE(GLOB COMMON_SRC_FILE
#   ${SAMPLE_PATH}/common/*.c
#   ${SAMPLE_PATH}/svp/common/*.c
# )


# 定义库
set(SYSTEM_LIBS pthread m dl stdc++)
set(OPEN_CV_LIBS ${OPENCV}/lib/libopencv_core.so ${OPENCV}/lib/libopencv_imgproc.so)  # Add as needed
set(SECURITY_LIBS ${MPP_PATH}/out/lib/libsecurec.a)
set(ACL_LIBS ${MPP_PATH}/out/lib/svp_npu/libsvp_acl.a ${MPP_PATH}/out/lib/svp_npu/libprotobuf-c.a)

FILE(GLOB SAMPLE_COMMON_SRC ${SAMPLE_PATH}/common/*.c)
FILE(GLOB SVP_COMMON_SRC ./src/svp/common/*.c)
FILE(GLOB IVE_SRC ./src/svp/ive/*.c)
add_executable(sample_ive_main ./src/examples/sample_ive_main.c ${SVP_COMMON_SRC} ${IVE_SRC} ${SAMPLE_COMMON_SRC})

add_executable(test_vio ./src/examples/test_vio.c ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC})
add_executable(test_vio_v2 ./src/examples/test_vio_v2.c ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC})
add_executable(test_vio_mono ./src/examples/test_vio_mono.c ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC})
add_executable(test_vio_mono_v2 ./src/examples/test_vio_mono_v2.c ${SVP_COMMON_SRC} ${SAMPLE_COMMON_SRC})

include(cmake/ive_link_libs.cmake)
# 链接库到目标
target_link_libraries(sample_ive_main
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)

target_link_libraries(test_vio
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)

target_link_libraries(test_vio_v2
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)

target_link_libraries(test_vio_mono
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)

target_link_libraries(test_vio_mono_v2
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)
