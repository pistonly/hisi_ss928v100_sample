#include "ot_common.h"
#include "ot_mipi_rx.h"
#include "ot_type.h"
#include "sample_comm.h"
#include "sample_common_ive.h"
#include "sample_common_svp.h"

#define VB_RAW_CNT_NONE 0
#define VB_LINEAR_RAW_CNT 5
#define VB_WDR_RAW_CNT 8
#define VB_MULTI_RAW_CNT 15
#define VB_YUV_ROUTE_CNT 10
#define VB_DOUBLE_YUV_CNT 15
#define VB_MULTI_YUV_CNT 30
#define SAMPLE_SVP_BLK_CNT 16

static sample_vi_cfg g_vi_config;
static ot_sample_svp_switch g_md_switch = {TD_FALSE, TD_TRUE};

static sample_vo_cfg g_vo_cfg = {
    .vo_dev = SAMPLE_VO_DEV_UHD,
    .vo_intf_type = OT_VO_INTF_HDMI,
    .intf_sync = OT_VO_OUT_1080P30,
    .bg_color = COLOR_RGB_BLACK,
    .pix_format = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420,
    .disp_rect = {0, 0, 1920, 1080},
    .image_size = {1920, 1080},
    .vo_part_mode = OT_VO_PARTITION_MODE_SINGLE,
    .dis_buf_len = 3, /* 3: def buf len for single */
    .dst_dynamic_range = OT_DYNAMIC_RANGE_SDR8,
    .vo_mode = VO_MODE_1MUX,
    .compress_mode = OT_COMPRESS_MODE_NONE,
};

static sample_comm_venc_chn_param g_venc_chn_param = {
    .frame_rate = 30, /* 30 is a number */
    .stats_time = 1,  /* 1 is a number */
    .gop = 30,        /* 30 is a number */
    .venc_size = {1920, 1080},
    .size = PIC_1080P,
    .profile = 0,
    .is_rcn_ref_share_buf = TD_FALSE,
    .gop_attr =
        {
            .gop_mode = OT_VENC_GOP_MODE_NORMAL_P,
            .normal_p = {2},
        },
    .type = OT_PT_H265,
    .rc_mode = SAMPLE_RC_VBR,
};

static td_void sample_vio_stop_vpss(ot_vpss_grp grp) {
  td_bool chn_enable[OT_VPSS_MAX_PHYS_CHN_NUM] = {TD_TRUE, TD_FALSE, TD_FALSE,
                                                  TD_FALSE};

  sample_common_vpss_stop(grp, chn_enable, OT_VPSS_MAX_PHYS_CHN_NUM);
}

static td_void sample_vio_stop_vo(td_void) {
  sample_comm_vo_stop_vo(&g_vo_cfg);
}

static td_s32
sample_common_svp_get_pic_type_by_sns_type(sample_sns_type sns_type,
                                           ot_pic_size size[], td_u32 num) {
  sample_svp_check_exps_return(
      num > OT_VPSS_CHN_NUM, TD_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR,
      "num(%u) can't be larger than (%u)\n", num, OT_VPSS_CHN_NUM);
  switch (sns_type) {
  case OV_OS08A20_MIPI_8M_30FPS_12BIT:
  case OV_OS08A20_MIPI_8M_30FPS_12BIT_WDR2TO1:
    size[0] = PIC_3840X2160;
    break;
  default:
    size[0] = PIC_3840X2160;
    break;
  }
  return TD_SUCCESS;
}

static td_s32 sample_common_svp_set_vi_cfg(sample_vi_cfg *vi_cfg,
                                           ot_pic_size *pic_type,
                                           td_u32 pic_type_len,
                                           ot_pic_size *ext_pic_size_type,
                                           sample_sns_type sns_type) {
  sample_comm_vi_get_default_vi_cfg(sns_type, vi_cfg);
  sample_svp_check_exps_return(pic_type_len < OT_VPSS_CHN_NUM, TD_FAILURE,
                               SAMPLE_SVP_ERR_LEVEL_ERROR,
                               "pic_type_len is illegal!\n");
  pic_type[1] = *ext_pic_size_type;

  return TD_SUCCESS;
}

static td_s32 sample_vio_start_venc(ot_venc_chn venc_chn[], td_u32 chn_num,
                                    const ot_size *in_size) {
  td_s32 i, ret;

  g_venc_chn_param.venc_size.width = in_size->width;
  g_venc_chn_param.venc_size.height = in_size->height;
  g_venc_chn_param.size = sample_comm_sys_get_pic_enum(in_size);

  for (i = 0; i < (td_s32)chn_num; i++) {
    ret = sample_comm_venc_start(venc_chn[i], &g_venc_chn_param);
    if (ret != TD_SUCCESS) {
      goto exit;
    }
  }

  ret = sample_comm_venc_start_get_stream(venc_chn, chn_num);
  if (ret != TD_SUCCESS) {
    goto exit;
  }

  return TD_SUCCESS;

exit:
  for (i = i - 1; i >= 0; i--) {
    sample_comm_venc_stop(venc_chn[i]);
  }
  return TD_FAILURE;
}

static td_void sample_vi_get_default_vb_config(ot_size *size, ot_vb_cfg *vb_cfg,
                                               ot_vi_video_mode video_mode,
                                               td_u32 yuv_cnt, td_u32 raw_cnt) {
  ot_vb_calc_cfg calc_cfg;
  ot_pic_buf_attr buf_attr;

  (td_void) memset_s(vb_cfg, sizeof(ot_vb_cfg), 0, sizeof(ot_vb_cfg));
  vb_cfg->max_pool_cnt = 128; /* 128 blks */

  /* default YUV pool: SP420 + compress_seg */
  buf_attr.width = size->width;
  buf_attr.height = size->height;
  buf_attr.align = OT_DEFAULT_ALIGN;
  buf_attr.bit_width = OT_DATA_BIT_WIDTH_8;
  buf_attr.pixel_format = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
  buf_attr.compress_mode = OT_COMPRESS_MODE_SEG;
  ot_common_get_pic_buf_cfg(&buf_attr, &calc_cfg);

  vb_cfg->common_pool[0].blk_size = calc_cfg.vb_size;
  vb_cfg->common_pool[0].blk_cnt = yuv_cnt;

  /* default raw pool: raw12bpp + compress_line */
  buf_attr.pixel_format = OT_PIXEL_FORMAT_RGB_BAYER_12BPP;
  buf_attr.compress_mode =
      (video_mode == OT_VI_VIDEO_MODE_NORM ? OT_COMPRESS_MODE_LINE
                                           : OT_COMPRESS_MODE_NONE);
  ot_common_get_pic_buf_cfg(&buf_attr, &calc_cfg);
  vb_cfg->common_pool[1].blk_size = calc_cfg.vb_size;
  vb_cfg->common_pool[1].blk_cnt = raw_cnt;
}

static td_s32 sample_vio_start_vpss(ot_vpss_grp grp, ot_size *in_size) {
  td_s32 ret;
  ot_low_delay_info low_delay_info;
  ot_vpss_grp_attr grp_attr;
  ot_vpss_chn_attr chn_attr;
  td_bool chn_enable[OT_VPSS_MAX_PHYS_CHN_NUM] = {TD_TRUE, TD_FALSE, TD_FALSE,
                                                  TD_FALSE};

  sample_comm_vpss_get_default_grp_attr(&grp_attr);
  grp_attr.max_width = in_size->width;
  grp_attr.max_height = in_size->height;
  sample_comm_vpss_get_default_chn_attr(&chn_attr);
  chn_attr.width = in_size->width;
  chn_attr.height = in_size->height;

  ret = sample_common_vpss_start(grp, chn_enable, &grp_attr, &chn_attr,
                                 OT_VPSS_MAX_PHYS_CHN_NUM);
  if (ret != TD_SUCCESS) {
    return ret;
  }

  low_delay_info.enable = TD_TRUE;
  low_delay_info.line_cnt = 200; /* 200: lowdelay line */
  low_delay_info.one_buf_en = TD_FALSE;
  ret = ss_mpi_vpss_set_low_delay_attr(grp, 0, &low_delay_info);
  if (ret != TD_SUCCESS) {
    sample_common_vpss_stop(grp, chn_enable, OT_VPSS_MAX_PHYS_CHN_NUM);
    return ret;
  }

  return TD_SUCCESS;
}

static td_s32 sample_common_svp_vb_init(ot_pic_size *pic_type,
                                        ot_size *pic_size,
                                        td_u32 vpss_chn_num) {
  td_s32 ret;
  td_u32 i;
  ot_vb_cfg vb_cfg = {0};
  ot_pic_buf_attr pic_buf_attr;
  ot_vb_calc_cfg calc_cfg;
  ot_vi_vpss_mode_type mode_type = OT_VI_ONLINE_VPSS_OFFLINE;
  ot_vi_video_mode video_mode = OT_VI_VIDEO_MODE_NORM;

  vb_cfg.max_pool_cnt = OT_SAMPLE_IVE_MAX_POOL_CNT;

  ret = sample_comm_sys_get_pic_size(pic_type[0], &pic_size[0]);
  sample_svp_check_exps_goto(
      ret != TD_SUCCESS, vb_fail_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
      "sample_comm_sys_get_pic_size failed,Error(%#x)!\n", ret);
  pic_buf_attr.width = pic_size[0].width;
  pic_buf_attr.height = pic_size[0].height;
  pic_buf_attr.align = OT_DEFAULT_ALIGN;
  pic_buf_attr.bit_width = OT_DATA_BIT_WIDTH_8;
  pic_buf_attr.pixel_format = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_422;
  pic_buf_attr.compress_mode = OT_COMPRESS_MODE_NONE;

  ot_common_get_pic_buf_cfg(&pic_buf_attr, &calc_cfg);

  vb_cfg.common_pool[0].blk_size = calc_cfg.vb_size;
  vb_cfg.common_pool[0].blk_cnt = SAMPLE_SVP_BLK_CNT;

  for (i = 1; (i < vpss_chn_num) && (i < OT_VB_MAX_COMMON_POOLS); i++) {
    ret = sample_comm_sys_get_pic_size(pic_type[i], &pic_size[i]);
    sample_svp_check_exps_goto(
        ret != TD_SUCCESS, vb_fail_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "sample_comm_sys_get_pic_size failed,Error(%#x)!\n", ret);
    pic_buf_attr.width = pic_size[i].width;
    pic_buf_attr.height = pic_size[i].height;
    pic_buf_attr.compress_mode = OT_COMPRESS_MODE_NONE;
    pic_buf_attr.align = OT_DEFAULT_ALIGN;

    ot_common_get_pic_buf_cfg(&pic_buf_attr, &calc_cfg);

    /* comm video buffer */
    vb_cfg.common_pool[i].blk_size = calc_cfg.vb_size;
    vb_cfg.common_pool[i].blk_cnt = SAMPLE_SVP_BLK_CNT;
  }

  ret = sample_comm_sys_init_with_vb_supplement(&vb_cfg,
                                                OT_VB_SUPPLEMENT_BNR_MOT_MASK);
  sample_svp_check_exps_goto(ret != TD_SUCCESS, vb_fail_1,
                             SAMPLE_SVP_ERR_LEVEL_ERROR,
                             "sample_comm_sys_init failed,Error(%#x)!\n", ret);

  ret = sample_comm_vi_set_vi_vpss_mode(mode_type, video_mode);
  sample_svp_check_exps_goto(ret != TD_SUCCESS, vb_fail_1,
                             SAMPLE_SVP_ERR_LEVEL_ERROR,
                             "sample_comm_vi_set_vi_vpss_mode failed!\n");
  return ret;
vb_fail_1:
  sample_comm_sys_exit();
vb_fail_0:
  return ret;
}

td_s32 main(td_s32 argc, td_char *argv[]) {
  td_s32 ret;
  ot_vi_vpss_mode_type mode_type = OT_VI_OFFLINE_VPSS_OFFLINE;
  ot_vi_video_mode video_mode = OT_VI_VIDEO_MODE_NORM;
  ot_vi_pipe vi_pipe[2] = {0, 1}; /* 2 pipe */
  const ot_vi_chn vi_chn = 0;
  ot_vpss_grp vpss_grp[2] = {0, 1}; /* 2 vpss grp */
  const td_u32 grp_num = 1;         /* 2 vpss grp */
  const ot_vpss_chn vpss_chn = 0;
  sample_vi_cfg vi_cfg[2];
  sample_sns_type sns_type = SENSOR0_TYPE;
  /* ot_size in_size; */
  ot_vb_cfg vb_cfg;

  ot_pic_size ext_pic_size_type = PIC_1080P;
  ot_size pic_size[OT_VPSS_CHN_NUM];
  ot_pic_size pic_type[OT_VPSS_CHN_NUM];

  ret = sample_common_svp_get_pic_type_by_sns_type(sns_type, pic_type,
                                                   OT_VPSS_CHN_NUM);
  ret = sample_common_svp_set_vi_cfg(vi_cfg, pic_type, OT_VPSS_CHN_NUM,
                                     &ext_pic_size_type, sns_type);
  ret = sample_common_svp_vb_init(pic_type, pic_size, OT_VPSS_CHN_NUM);

  ret = sample_comm_vi_set_vi_vpss_mode(mode_type, video_mode);
  if (ret != TD_SUCCESS) {
    sample_svp_trace_err("sample_comm_vi_set_vi_vpss_mode!");
    return 1;
  }

  // these two lines are important
  vi_cfg[0].mipi_info.divide_mode = LANE_DIVIDE_MODE_1;
  vi_cfg[0].sns_info.bus_id = 5;

  // start vi vpss
  td_s32 dev_num = 1;
  td_s32 i, j;
  for (i = 0; i < dev_num; i++) {
    ret = sample_comm_vi_start_vi(&vi_cfg[i]);
    if (ret != TD_SUCCESS) {
      sample_svp_trace_err("sample_comm_vi_start_vi error, dev_id: %d", i);
      goto start_vi_failed;
    }
  }

  for (i = 0; i < grp_num; i++) {
    sample_comm_vi_bind_vpss(vi_cfg[i].bind_pipe.pipe_id[0], 0, vpss_grp[i], 0);
  }

  for (i = 0; i < grp_num; i++) {
    ret = sample_vio_start_vpss(vpss_grp[i], pic_size);
    if (ret != TD_SUCCESS) {
      sample_svp_trace_err("sample_vio_start_vpss error, dev_id: %d", i);
      goto start_vpss_failed;
    }
  }

  // start venc vo
  sample_vo_mode vo_mode = VO_MODE_1MUX;
  const ot_vo_layer vo_layer = 0;
  ot_vo_chn vo_chn[4] = {0, 1, 2, 3};     /* 4: max chn num, 0/1/2/3 chn id */
  ot_venc_chn venc_chn[4] = {0, 1, 2, 3}; /* 4: max chn num, 0/1/2/3 chn id */

  if (grp_num > 1) {
    vo_mode = VO_MODE_4MUX;
  }

  g_vo_cfg.vo_mode = vo_mode;
  ret = sample_comm_vo_start_vo(&g_vo_cfg);
  if (ret != TD_SUCCESS) {
    goto start_vo_failed;
  }

  ret = sample_vio_start_venc(venc_chn, grp_num, pic_size);
  if (ret != TD_SUCCESS) {
    goto start_venc_failed;
  }

  for (i = 0; i < grp_num; i++) {
    sample_comm_vpss_bind_vo(vpss_grp[i], vpss_chn, vo_layer, vo_chn[i]);
    sample_comm_vpss_bind_venc(vpss_grp[i], vpss_chn, venc_chn[i]);
  }
  return TD_SUCCESS;

start_venc_failed:
  sample_vio_stop_vo();
start_vo_failed:
  return TD_FAILURE;

start_vpss_failed:
  for (j = i - 1; j >= 0; j--) {
    sample_vio_stop_vpss(vpss_grp[j]);
  }

  for (i = 0; i < grp_num; i++) {
    sample_comm_vi_un_bind_vpss(i, 0, vpss_grp[i], 0);
  }
start_vi_failed:
  for (j = i - 1; j >= 0; j--) {
    sample_comm_vi_stop_vi(&vi_cfg[j]);
  }
  return 1;
}
