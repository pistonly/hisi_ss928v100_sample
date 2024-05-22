//
// Created by SOUMA on 2024/4/30.
//

#ifndef PROFILE_PY_SAMPLE_VIO_H
#define PROFILE_PY_SAMPLE_VIO_H

#ifdef __cplusplus
extern "C" {
#endif /* end of #ifdef __cplusplus */
#include "sample_comm.h"
    void set_vo_4k60();

    hi_s32 sample_vio_sys_init(hi_vi_vpss_mode_type mode_type, hi_vi_video_mode video_mode,
                           hi_u32 yuv_cnt, hi_u32 raw_cnt);
    hi_void sample_vi_get_two_sensor_vi_cfg(sample_sns_type sns_type, sample_vi_cfg *vi_cfg0, sample_vi_cfg *vi_cfg1);

hi_s32 sample_vio_start_multi_vi_vpss(sample_vi_cfg *vi_cfg, hi_vpss_grp *vpss_grp,
                                      hi_s32 dev_num, hi_s32 grp_num);
hi_s32 sample_vio_start_venc_and_vo(hi_vpss_grp vpss_grp[], hi_u32 grp_num, const hi_size *in_size);
hi_void sample_get_char(hi_void);
hi_void sample_vio_stop_venc_and_vo(hi_vpss_grp vpss_grp[], hi_u32 grp_num);
hi_void sample_vio_stop_vpss(hi_vpss_grp grp);
hi_s32 get_frame_from_vpss_grp(hi_vpss_grp vpss_grp[], hi_s32 grp_num, uint8_t* yuv_planner);

#ifdef __cplusplus
}
#endif /* end of #ifdef __cplusplus */

#endif //PROFILE_PY_SAMPLE_VIO_H
