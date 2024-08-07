#include "ot_common_vdec.h"
#include "ot_defines.h"
#include "ot_type.h"
#include "sample_comm.h"
extern "C" {
#include "ot_common_video.h"
#include <libavformat/avformat.h>
}

#define UHD_STREAM_WIDTH 3840
#define UHD_STREAM_HEIGHT 2160
#define FHD_STREAM_WIDTH 1920
#define FHD_STREAM_HEIGHT 1080
#define REF_NUM 2
#define DISPLAY_NUM 2
#define SAMPLE_VDEC_COMM_VB_CNT 4
#define SAMPLE_VDEC_VPSS_LOW_DELAY_LINE_CNT 16

static ot_payload_type g_cur_type = OT_PT_H264;

static vdec_display_cfg g_vdec_display_cfg = {
    .pic_size = PIC_3840X2160,
    .intf_sync = OT_VO_OUT_3840x2160_30,
    .intf_type = OT_VO_INTF_HDMI,
};

static ot_size g_disp_size;
static td_s32 g_sample_exit = 0;

static td_u32 sample_vdec_get_dimension(bool is_width) {
  if (g_cur_type == OT_PT_H264 || g_cur_type == OT_PT_H265 ||
      g_cur_type == OT_PT_JPEG || g_cur_type == OT_PT_MJPEG) {
    return is_width ? UHD_STREAM_WIDTH : UHD_STREAM_HEIGHT;
  }
  sample_print("Invalid type %d!\n", g_cur_type);
  return is_width ? UHD_STREAM_WIDTH : UHD_STREAM_HEIGHT;
}

static td_s32 sample_start_vdec(sample_vdec_attr *sample_vdec,
                                td_u32 vdec_chn_num, td_u32 len) {
  td_s32 ret = sample_comm_vdec_start(vdec_chn_num, sample_vdec, len);
  if (ret != TD_SUCCESS) {
    sample_print("Start VDEC failed for %#x!\n", ret);
    sample_comm_vdec_stop(vdec_chn_num);
  }
  return ret;
}

static td_s32 sample_init_module_vb(sample_vdec_attr *sample_vdec,
                                    td_u32 vdec_chn_num, ot_payload_type type,
                                    td_u32 len) {
  for (td_u32 i = 0; i < vdec_chn_num && i < len; i++) {
    sample_vdec[i].type = type;
    sample_vdec[i].width = sample_vdec_get_dimension(true);
    sample_vdec[i].height = sample_vdec_get_dimension(false);
    sample_vdec[i].mode = sample_comm_vdec_get_lowdelay_en()
                              ? OT_VDEC_SEND_MODE_COMPAT
                              : OT_VDEC_SEND_MODE_FRAME;
    sample_vdec[i].sample_vdec_video.dec_mode = OT_VIDEO_DEC_MODE_IP;
    sample_vdec[i].sample_vdec_video.bit_width = OT_DATA_BIT_WIDTH_8;
    sample_vdec[i].sample_vdec_video.ref_frame_num = REF_NUM;
    sample_vdec[i].display_frame_num = DISPLAY_NUM;
    sample_vdec[i].frame_buf_cnt =
        (type == OT_PT_JPEG) ? (sample_vdec[i].display_frame_num + 1)
                             : (sample_vdec[i].sample_vdec_video.ref_frame_num +
                                sample_vdec[i].display_frame_num + 1);
    if (type == OT_PT_JPEG) {
      sample_vdec[i].sample_vdec_picture.pixel_format =
          OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
      sample_vdec[i].sample_vdec_picture.alpha = 255; // Alpha value
    }
  }
  td_s32 ret = sample_comm_vdec_init_vb_pool(vdec_chn_num, sample_vdec, len);
  if (ret != TD_SUCCESS) {
    sample_print("Init module VB failed for %#x!\n", ret);
    return ret;
  }
  return TD_SUCCESS;
}

static td_s32 sample_init_sys_and_vb(sample_vdec_attr *sample_vdec,
                                     td_u32 vdec_chn_num, ot_payload_type type,
                                     td_u32 len) {
  ot_vb_cfg vb_cfg;
  ot_pic_buf_attr buf_attr = {0};
  td_s32 ret;

  ret = sample_comm_sys_get_pic_size(g_vdec_display_cfg.pic_size, &g_disp_size);
  if (ret != TD_SUCCESS) {
    sample_print("System get picture size failed for %#x!\n", ret);
    return ret;
  }
  buf_attr.align = OT_DEFAULT_ALIGN;
  buf_attr.bit_width = OT_DATA_BIT_WIDTH_8;
  buf_attr.compress_mode = OT_COMPRESS_MODE_SEG;
  buf_attr.height = g_disp_size.height;
  buf_attr.width = g_disp_size.width;
  buf_attr.pixel_format = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420;

  memset_s(&vb_cfg, sizeof(vb_cfg), 0, sizeof(vb_cfg));
  vb_cfg.max_pool_cnt = 1;
  vb_cfg.common_pool[0].blk_cnt = SAMPLE_VDEC_COMM_VB_CNT * vdec_chn_num;
  vb_cfg.common_pool[0].blk_size = ot_common_get_pic_buf_size(&buf_attr);
  ret = sample_comm_sys_init(&vb_cfg);
  if (ret != TD_SUCCESS) {
    sample_print("System init failed for %#x!\n", ret);
    sample_comm_sys_exit();
    return ret;
  }
  ret = sample_init_module_vb(sample_vdec, vdec_chn_num, type, len);
  if (ret != TD_SUCCESS) {
    sample_print("Module VB init failed for %#x!\n", ret);
    sample_comm_vdec_exit_vb_pool();
    sample_comm_sys_exit();
    return ret;
  }
  return TD_SUCCESS;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <RTSP URL>\n", argv[0]);
    return -1;
  }

  const char *rtsp_url = argv[1];
  avformat_network_init();

  AVFormatContext *fmt_ctx = NULL;
  if (avformat_open_input(&fmt_ctx, rtsp_url, NULL, NULL) != 0) {
    fprintf(stderr, "Could not open source\n");
    return -1;
  }

  if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
    fprintf(stderr, "Could not find stream information\n");
    return -1;
  }

  int video_stream_index = -1;
  for (int i = 0; i < fmt_ctx->nb_streams; i++) {
    if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_index = i;
      break;
    }
  }

  if (video_stream_index == -1) {
    fprintf(stderr, "Could not find video stream\n");
    return -1;
  }

  td_s32 ret;
  td_u32 vdec_chn_num = 1;
  sample_vdec_attr sample_vdec[OT_VDEC_MAX_CHN_NUM];

  ret = sample_init_sys_and_vb(sample_vdec, vdec_chn_num, g_cur_type,
                               OT_VDEC_MAX_CHN_NUM);
  if (ret != TD_SUCCESS) {
    return ret;
  }
  fprintf(stdout, "vdec_ch_num: %d\n", vdec_chn_num);

  ret = sample_start_vdec(sample_vdec, vdec_chn_num, OT_VDEC_MAX_CHN_NUM);
  if (ret != TD_SUCCESS) {
    fprintf(stderr, "Failed to start VDEC\n");
    return -1;
  }

  AVPacket packet;
  ot_vdec_stream stream;
  int packet_num = 0;
  ot_vdec_chn_status vdec_status;

  while (av_read_frame(fmt_ctx, &packet) >= 0) {
    if (packet.stream_index == video_stream_index) {
      stream.addr = packet.data;
      stream.len = packet.size;
      stream.pts = packet.pts;
      stream.need_display = TD_TRUE;
      stream.end_of_frame = (packet_num > 1000) ? TD_TRUE : TD_FALSE;
      stream.end_of_stream =
          (packet.flags & AV_PKT_FLAG_KEY) ? TD_TRUE : TD_FALSE;

      fprintf(stdout, "Sending stream to decoder, packet_num: %d\n",
              packet_num++);
      ret = ss_mpi_vdec_send_stream(0, &stream, -1);
      if (ret != TD_SUCCESS) {
        fprintf(stderr, "Error sending stream to decoder for %#x!\n", ret);
        break;
      }
    }
    av_packet_unref(&packet);

    ret = ss_mpi_vdec_query_status(0, &vdec_status);
    if (ret != TD_SUCCESS) {
      fprintf(stderr, "Error querying VDEC status!\n");
    } else {
      fprintf(stdout,
              "INFO: \n type: %d, left bytes: %d, left frames: %d, left "
              "decoded_frames: %d, is_started: %d, recv_stream_frames: %d, "
              "dec_stream_frames: %d, dec_w: %d, dec_h: %d\n",
              vdec_status.type, vdec_status.left_stream_bytes,
              vdec_status.left_stream_frames, vdec_status.left_decoded_frames,
              vdec_status.is_started, vdec_status.recv_stream_frames,
              vdec_status.dec_stream_frames, vdec_status.width,
              vdec_status.height);
      fprintf(
          stdout,
          "VDEC status error: \n set_pic_size_err: %d, set_protocol_num_err: "
          "%d, set_ref_num_err: %d, set_pic_buf_size_err: %d, format_err: %d, "
          "stream_unsupport: %d, pack_err: %d, stream_size_over: %d, stream "
          "not release: %d\n",
          vdec_status.dec_err.set_pic_size_err,
          vdec_status.dec_err.set_protocol_num_err,
          vdec_status.dec_err.set_ref_num_err,
          vdec_status.dec_err.set_pic_buf_size_err,
          vdec_status.dec_err.format_err, vdec_status.dec_err.stream_unsupport,
          vdec_status.dec_err.pack_err, vdec_status.dec_err.stream_size_over,
          vdec_status.dec_err.stream_not_release);

      if (vdec_status.left_decoded_frames > 1) {
        ot_video_frame_info frame;
        ret = ss_mpi_vdec_get_frame(0, &frame, NULL, 100);
        if (ret != TD_SUCCESS) {
          fprintf(stderr, "Error getting frame %#x!\n", ret);
        } else {
          fprintf(stdout, "Out: frame_w: %d\n", frame.video_frame.width);
          ret = ss_mpi_vdec_release_frame(0, &frame);
          if (ret != TD_SUCCESS) {
            fprintf(stderr, "Error releasing frame %#x!\n", ret);
          }
        }
      }
    }
  }

  sample_comm_vdec_stop(1);
  sample_comm_vdec_exit_vb_pool();
  sample_comm_sys_exit();
  avformat_close_input(&fmt_ctx);
  avformat_network_deinit();

  return 0;
}
