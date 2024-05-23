/*
  Copyright (c), 2001-2022, Shenshu Tech. Co., Ltd.
 */

#ifndef SAMPLE_IVE_MAIN_H
#define SAMPLE_IVE_MAIN_H
#include "ot_type.h"
#include "securec.h"

/*
 * function : show Canny sample
 */
td_void sample_ive_canny(td_char canny_complete);

/*
 * function : show Gmm2 sample
 */
td_void sample_ive_gmm2(td_void);

/*
 * function : show Test Memory sample
 */
td_void sample_ive_test_memory(td_void);

/*
 * function : show Sobel sample
 */
td_void sample_ive_sobel(td_void);

/*
 * function : show St Lk sample
 */
td_void sample_ive_st_lk(td_void);

/*
 * function : show Occlusion detected sample
 */
td_void sample_ive_od(td_void);

/*
 * function : show Md sample
 */
td_void sample_ive_md(td_void);

/*
 * function : show PerspTrans sample
 */
td_void sample_ive_persp_trans(td_void);

/*
 * function :Canny sample signal handle
 */
td_void sample_ive_canny_handle_sig(td_void);

/*
 * function : Gmm2 sample signal handle
 */
td_void sample_ive_gmm2_handle_sig(td_void);


/*
 * function : TestMemory sample signal handle
 */
td_void sample_ive_test_memory_handle_sig(td_void);

/*
 * function : Sobel sample signal handle
 */
td_void sample_ive_sobel_handle_sig(td_void);

/*
 * function : St_Lk sample signal handle
 */
td_void sample_ive_st_lk_handle_sig(td_void);

/*
 * function : PerspTrans sample signal handle
 */
td_void sample_ive_persp_trans_handle_sig(td_void);

/*
 * function : Od sample signal handle
 */
td_void sample_ive_od_handle_sig(td_void);

/*
 * function : Md sample signal handle
 */
td_void sample_ive_md_handle_sig(td_void);

#endif

