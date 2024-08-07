FILE(GLOB VDEC_SRC ./*.c)
add_executable(sample_vdec_ly sample_vdec.c ${SAMPLE_COMMON_SRC})

add_executable(sample_vdec_save sample_vdec_save.c ${SAMPLE_COMMON_SRC})

add_executable(ffmpeg_vdec_vpss ffmpeg_vdec_vpss.cpp ${SAMPLE_COMMON_SRC})
# add_executable(ffmpeg_vdec ffmpeg_vdec.cpp ${SAMPLE_COMMON_SRC})


target_link_libraries(sample_vdec_ly
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)

target_link_libraries(sample_vdec_save
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${IVE_LINK_LIBS}
  stdc++ # note this line is important
)

target_link_libraries(ffmpeg_vdec_vpss
  ${SYSTEM_LIBS}
  ${SECURITY_LIBS}
  ${IVE_LINK_LIBS}
  avformat
  avcodec
  avutil
  swresample
  swscale
  stdc++ # note this line is important
)


# target_link_libraries(ffmpeg_vdec
#   ${SYSTEM_LIBS}
#   ${SECURITY_LIBS}
#   ${IVE_LINK_LIBS}
#   avformat
#   avcodec
#   avutil
#   swresample
#   swscale
#   stdc++ # note this line is important
# )

# target_link_libraries(ffmpeg_vdec_vpss
#   avformat
#   avcodec
#   avutil
#   swresample
#   swscale
#   ${SYSTEM_LIBS}
#   ${SECURITY_LIBS}
#   ${IVE_LINK_LIBS}
#   stdc++ 
# )
