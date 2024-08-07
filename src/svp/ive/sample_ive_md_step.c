
#include "ot_common_ive.h"
#include "ot_common_svp.h"
#include "ot_type.h"
#include "sample_common_ive.h"
#include "sample_common_svp.h"
#include "ss_mpi_ive.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define OT_SAMPLE_MD_SRC_NUM 2
#define OT_SAMPLE_MD_WIDTH 1920
#define OT_SAMPLE_MD_HEIGHT 1080


typedef struct {
  ot_svp_src_img img[OT_SAMPLE_MD_SRC_NUM];
  ot_svp_dst_img diff;
  ot_svp_src_img sad_zeros;
  ot_svp_dst_img sad;
  ot_svp_dst_img sad_thres;
  ot_svp_dst_mem_info blob;
} ot_sample_MDStep_info;

td_void sample_ive_md_step(td_void) {
  td_s32 ret;
  /* open input file */
  const td_char *src_file = "./data/input/md/md_1920x1080.bin";
  /* const td_char *src_file = "./data/input/md/fake_1920x1080_50.bin"; */
  /* const td_char *src_file = "./data/input/md/fake_1920x1080_20.bin"; */
  /* const td_char *src_file = "./data/input/md/fake_1920x1080_255.bin"; */
  /* const td_char *src_file = "./data/input/md/fake_1920x1080_230.bin"; */
  td_char path[PATH_MAX] = {0};
  sample_svp_check_exps_return(
      (strlen(src_file) > PATH_MAX) || (realpath(src_file, path) == TD_NULL),
      OT_ERR_IVE_ILLEGAL_PARAM, SAMPLE_SVP_ERR_LEVEL_ERROR, "invalid file!\n");
  FILE *fp_src = fopen(path, "rb");
  sample_svp_check_exps_return(fp_src == TD_NULL, OT_ERR_IVE_ILLEGAL_PARAM,
                               SAMPLE_SVP_ERR_LEVEL_ERROR,
                               "Open file failed!\n");
  /* open output file */
  sample_svp_check_exps_return((realpath("./data/output/md", path) == TD_NULL),
                               OT_ERR_IVE_ILLEGAL_PARAM,
                               SAMPLE_SVP_ERR_LEVEL_ERROR, "invalid dir!\n");
  ret = strcat_s(path, PATH_MAX, "/1920x1080.yuv");
  FILE *fp_out = fopen(path, "wb");
  sample_svp_check_exps_return(fp_out == TD_NULL, OT_ERR_IVE_ILLEGAL_PARAM,
                               SAMPLE_SVP_ERR_LEVEL_ERROR,
                               "Open file failed!\n");

  sample_svp_check_exps_return((realpath("./data/output/md", path) == TD_NULL),
                               OT_ERR_IVE_ILLEGAL_PARAM,
                               SAMPLE_SVP_ERR_LEVEL_ERROR, "invalid dir!\n");
  ret = strcat_s(path, PATH_MAX, "/diff.yuv");
  FILE *fp_diff = fopen(path, "wb");
  sample_svp_check_exps_return(fp_diff == TD_NULL, OT_ERR_IVE_ILLEGAL_PARAM,
                               SAMPLE_SVP_ERR_LEVEL_ERROR,
                               "Open file failed!\n");

  sample_svp_check_exps_return((realpath("./data/output/md", path) == TD_NULL),
                               OT_ERR_IVE_ILLEGAL_PARAM,
                               SAMPLE_SVP_ERR_LEVEL_ERROR, "invalid dir!\n");
  ret = strcat_s(path, PATH_MAX, "/sad.yuv");
  FILE *fp_sad = fopen(path, "wb");
  sample_svp_check_exps_return(fp_sad == TD_NULL, OT_ERR_IVE_ILLEGAL_PARAM,
                               SAMPLE_SVP_ERR_LEVEL_ERROR,
                               "Open sad file failed!\n");

  sample_svp_check_exps_return((realpath("./data/output/md", path) == TD_NULL),
                               OT_ERR_IVE_ILLEGAL_PARAM,
                               SAMPLE_SVP_ERR_LEVEL_ERROR, "invalid dir!\n");
  ret = strcat_s(path, PATH_MAX, "/sad_thred.yuv");
  FILE *fp_sad_thred = fopen(path, "wb");
  sample_svp_check_exps_return(fp_sad_thred == TD_NULL, OT_ERR_IVE_ILLEGAL_PARAM,
                               SAMPLE_SVP_ERR_LEVEL_ERROR,
                               "Open sad_thred file failed!\n");

  /* open output rois file */
  sample_svp_check_exps_return((realpath("./data/output/md", path) == TD_NULL),
                               OT_ERR_IVE_ILLEGAL_PARAM,
                               SAMPLE_SVP_ERR_LEVEL_ERROR, "invalid dir!\n");
  ret = strcat_s(path, PATH_MAX, "/rois.bin");
  FILE *fp_out_rois = fopen(path, "wb");
  sample_svp_check_exps_return(fp_out_rois == TD_NULL, OT_ERR_IVE_ILLEGAL_PARAM,
                               SAMPLE_SVP_ERR_LEVEL_ERROR,
                               "Open file failed!\n");

  td_s32 img_num = 0;
  td_u8 rois[32 * 32 * OT_SVP_RECT_NUM];

  ot_sample_MDStep_info mdstep;

  ret = sample_common_ive_check_mpi_init();
  sample_svp_check_exps_return_void(ret != TD_TRUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                    "ive_check_mpi_init failed!\n");

  // md init
  (td_void)memset_s(&mdstep, sizeof(ot_sample_MDStep_info), 0, sizeof(ot_sample_MDStep_info));
  for (td_u16 i=0; i < OT_SAMPLE_MD_SRC_NUM; i++) {
    ret = sample_common_ive_create_image(&(mdstep.img[i]), OT_SVP_IMG_TYPE_U8C1, OT_SAMPLE_MD_WIDTH, OT_SAMPLE_MD_HEIGHT);
    sample_svp_check_exps_return(
        ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),Create img[%d] image failed!\n", ret, i);
  }
  ret = sample_common_ive_create_image(&mdstep.diff, OT_SVP_IMG_TYPE_U8C1,
                                       OT_SAMPLE_MD_WIDTH,
                                       OT_SAMPLE_MD_HEIGHT);
  sample_svp_check_exps_return(ret != TD_SUCCESS, ret,
                               SAMPLE_SVP_ERR_LEVEL_ERROR,
                               "Error(%#x),Create diff image failed!\n", ret);

  ret = sample_common_ive_create_image(&mdstep.sad_zeros, OT_SVP_IMG_TYPE_U8C1,
                                       OT_SAMPLE_MD_WIDTH, OT_SAMPLE_MD_HEIGHT);

  ret = sample_common_ive_create_image(&mdstep.sad, OT_SVP_IMG_TYPE_U16C1,
                                       OT_SAMPLE_MD_WIDTH / 4,
                                       OT_SAMPLE_MD_HEIGHT / 4);

  ret = sample_common_ive_create_image(&mdstep.sad_thres, OT_SVP_IMG_TYPE_U8C1,
                                       OT_SAMPLE_MD_WIDTH / 4,
                                       OT_SAMPLE_MD_HEIGHT / 4);

  td_u32 size = sizeof(ot_ive_ccblob);
  ret = sample_common_ive_create_mem_info(&mdstep.blob, size);
  sample_svp_check_exps_goto(ret != TD_SUCCESS, fail,  SAMPLE_SVP_ERR_LEVEL_ERROR,
                             "Error(%#x),Create blob mem info failed!\n", ret);

  ot_ive_sub_ctrl sub_ctrl;
  sub_ctrl.mode = OT_IVE_SUB_MODE_ABS;
  ot_ive_handle handle;

  ot_ive_threshold_ctrl thre_ctrl;
  thre_ctrl.mode = OT_IVE_THRESHOLD_MODE_BINARY;
  thre_ctrl.low_threshold = 15;
  thre_ctrl.min_val = 0;
  thre_ctrl.max_val = 255;

  ot_ive_ccl_ctrl ccl_ctrl;
  ccl_ctrl.mode = OT_IVE_CCL_MODE_4C;
  ccl_ctrl.init_area_threshold = 0;
  ccl_ctrl.step = 10;

  ot_ive_sad_ctrl sad_ctrl;
  sad_ctrl.mode = OT_IVE_SAD_MODE_MB_4X4;
  sad_ctrl.out_ctrl = OT_IVE_SAD_OUT_CTRL_16BIT_BOTH;
  sad_ctrl.max_val = 255;
  sad_ctrl.min_val = 0;
  sad_ctrl.threshold = 100;

  // process
  ret = sample_common_ive_init_zeros_img(&mdstep.sad_zeros);
  td_u16 roi_num = 0;

  for (td_u16 i = 0; i < 30; i++) {
    sample_svp_trace_debug("img_id: %d\n", i);
    td_u16 current_img_i = i % 2;
    ret = sample_common_ive_read_file(&mdstep.img[current_img_i], fp_src);
    sample_svp_check_exps_goto(ret != TD_SUCCESS, fail,
                               SAMPLE_SVP_ERR_LEVEL_ERROR,
                               "Error(%#x),Read src file failed!\n", ret);
    if (i == 0)
      continue;

    ret = ss_mpi_ive_sub(&handle, &mdstep.img[current_img_i],
                         &mdstep.img[1 - current_img_i], &mdstep.diff,
                         &sub_ctrl, TD_TRUE);
    sample_svp_check_exps_return(ret != TD_SUCCESS, TD_FALSE,
                                 SAMPLE_SVP_ERR_LEVEL_ERROR,
                                 "Error(%#x),ss_mpi_ive_sub failed!\n", ret);

    ret = ss_mpi_ive_threshold(&handle, &mdstep.diff, &mdstep.diff,
                               &thre_ctrl,
                               TD_TRUE);

    ret = ss_mpi_ive_sad(&handle, &mdstep.diff,
                         &mdstep.sad_zeros, &mdstep.sad, &mdstep.sad_thres, &sad_ctrl, TD_TRUE);

    ret = ss_mpi_ive_ccl(&handle, &mdstep.sad_thres, &mdstep.blob, &ccl_ctrl, TD_TRUE);
    sample_svp_check_exps_goto(
        ret != TD_SUCCESS, fail, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x), ss_mpi_ive_ccl failed!\n", ret);

    sample_common_ive_blob_to_rois(
        sample_svp_convert_addr_to_ptr(ot_ive_ccblob, mdstep.blob.virt_addr),
        &mdstep.img[current_img_i], OT_SVP_RECT_NUM, 10, rois, &roi_num, 4, 4);

    // save diff
    ret = sample_common_ive_write_file(&mdstep.diff, fp_diff);
    sample_svp_check_exps_goto(
        ret != TD_SUCCESS, fail, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),sample_common_ive_write failed!\n", ret);

    ret = sample_common_ive_write_file(&mdstep.sad, fp_sad);
    sample_svp_check_exps_goto(
        ret != TD_SUCCESS, fail, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),sample_common_ive_write failed!\n", ret);

    // save sad thred
    ret = sample_common_ive_write_file(&mdstep.sad_thres, fp_sad_thred);
    sample_svp_check_exps_goto(
        ret != TD_SUCCESS, fail, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),sample_common_ive_write failed!\n", ret);

    fwrite(&rois, 32 * 32 * roi_num, 1, fp_out_rois);
  }

fail:
  for (td_u16 i = 0; i < OT_SAMPLE_MD_SRC_NUM; i++) {
    sample_svp_mmz_free(mdstep.img[i].phys_addr[0], mdstep.img[i].virt_addr[0]);
  }
  sample_svp_mmz_free(mdstep.blob.phys_addr, mdstep.blob.virt_addr);
  sample_svp_close_file(fp_src);
  sample_svp_close_file(fp_out);
  sample_svp_close_file(fp_sad);
  sample_svp_close_file(fp_sad_thred);
  sample_svp_close_file(fp_diff);
  sample_svp_close_file(fp_out_rois);
}
