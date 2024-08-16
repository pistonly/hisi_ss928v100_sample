import cv2
import numpy as np
import re

# 文件路径
path = "/home/liuyang/Documents/haisi/my_samples/output/frame3840x2160_0.yuv"

# YUV 图像的宽度和高度（假设是 3840x2160 的 4K 图像）
match = re.search(r'(\d+)x(\d+)', path)
if match:
    width = int(match.group(1))
    height = int(match.group(2))
else:
    width = 3840
    height = 1080

# 读取 YUV 文件
with open(path, 'rb') as f:
    yuv_data = np.frombuffer(f.read(), dtype=np.uint8)

# 将 YUV 数据重塑为 YUV420 格式的图像
yuv_image = yuv_data.reshape((height * 3 // 2, width))
# yuv_image = yuv_data.reshape((height, width))


# 将 YUV 图像转换为 BGR 格式
bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
# bgr_image = yuv_image

# 将 BGR 图像保存为 JPEG 文件
output_path = f"/home/liuyang/Documents/haisi/my_samples/output/bgr{width}x{height}_0.jpg"
cv2.imwrite(output_path, bgr_image)

print(f"BGR image saved to {output_path}")
