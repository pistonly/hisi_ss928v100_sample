#include "ot_type.h"
#include "sample_common_svp.h"

static sample_vi_cfg g_vi_config;
static ot_sample_svp_switch g_md_switch = {TD_FALSE, TD_TRUE};

td_s32 main(td_s32 argc, td_char *argv[]) {
  td_s32 ret;
  ot_pic_size pic_type = PIC_1080P;
  ot_size pic_size;

  ret = sample_common_svp_start_vi_vpss_venc_vo(&g_vi_config, &g_md_switch,
                                                &pic_type);
  sample_svp_check_exps_goto(
      ret != TD_SUCCESS, end_md_0, sample_svp_err_level_error,
      "Error(%#x), sample_comm_svp_start_vi_vpss_venc_vo failed!\n", ret);

  ret = sample_comm_sys_get_pic_size(pic_type, &pic_size);
  sample_svp_check_exps_goto(
      ret != TD_SUCCESS, end_md_0, sample_svp_err_level_error,
      "Error(%#x),sample_comm_sys_get_pic_size failed!\n", ret);

end_md_0:
  return 1;
}
