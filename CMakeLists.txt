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

# 链接库到目标
target_link_libraries(sample_ive_main
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${ACL_LIBS}
  ${MPP_PATH}/out/lib/libss_bnr.a
  ${MPP_PATH}/out/lib/libss_ir_auto.a
  ${MPP_PATH}/out/lib/libss_upvqe.a
  ${MPP_PATH}/out/lib/libvqe_talkv2.a
  ${MPP_PATH}/out/lib/libss_acs.a
  ${MPP_PATH}/out/lib/libsns_os05a10_2l_slave.a
  ${MPP_PATH}/out/lib/libss_ae.a
  ${MPP_PATH}/out/lib/libss_isp.a
  ${MPP_PATH}/out/lib/libss_ldci.a
  ${MPP_PATH}/out/lib/libaac_sbr_dec.a
  ${MPP_PATH}/out/lib/libvqe_agc.a
  ${MPP_PATH}/out/lib/libsns_os08a20.a
  ${MPP_PATH}/out/lib/libss_mau.a
  ${MPP_PATH}/out/lib/libvqe_res.a
  ${MPP_PATH}/out/lib/libvqe_record.a
  ${MPP_PATH}/out/lib/libss_voice_engine.a
  ${MPP_PATH}/out/lib/libss_drc.a
  ${MPP_PATH}/out/lib/libsns_imx347_slave.a
  ${MPP_PATH}/out/lib/libss_md.a
  ${MPP_PATH}/out/lib/libss_dnvqe.a
  ${MPP_PATH}/out/lib/libsns_os08b10.a
  ${MPP_PATH}/out/lib/libsns_os04a10.a
  ${MPP_PATH}/out/lib/libsns_imx485.a
  ${MPP_PATH}/out/lib/libss_dpu_match.a
  ${MPP_PATH}/out/lib/libss_hdmi.a
  ${MPP_PATH}/out/lib/libss_extend_stats.a
  ${MPP_PATH}/out/lib/libss_awb.a
  ${MPP_PATH}/out/lib/libaac_sbr_enc.a
  ${MPP_PATH}/out/lib/libss_dsp.a
  ${MPP_PATH}/out/lib/libvqe_eq.a
  ${MPP_PATH}/out/lib/libss_dpu_rect.a
  ${MPP_PATH}/out/lib/libot_isp.a
  ${MPP_PATH}/out/lib/libvqe_anr.a
  ${MPP_PATH}/out/lib/libss_ive.a
  ${MPP_PATH}/out/lib/libaac_enc.a
  ${MPP_PATH}/out/lib/libsecurec.a
  ${MPP_PATH}/out/lib/libss_dehaze.a
  ${MPP_PATH}/out/lib/libss_crb.a
  ${MPP_PATH}/out/lib/libvqe_hpf.a
  ${MPP_PATH}/out/lib/libaac_dec.a
  ${MPP_PATH}/out/lib/libss_calcflicker.a
  ${MPP_PATH}/out/lib/libvqe_aec.a
  ${MPP_PATH}/out/lib/libvqe_wnr.a
  ${MPP_PATH}/out/lib/libaac_comm.a
  ${MPP_PATH}/out/lib/libss_mpi.a
  stdc++ # note this line is important
)



