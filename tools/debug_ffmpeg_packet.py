import struct
import numpy as np

def parse_packets(file_path):
    with open(file_path, 'rb') as f:
        packet_count = 0
        while True:
            # 读取4个字节的长度
            length_data = f.read(4)
            if len(length_data) < 4:
                break  # 如果不足4个字节，说明文件结束
            # 解析长度数据（假设是一个32位无符号整数）
            packet_length = struct.unpack('I', length_data)[0]

            # 读取数据包内容
            packet_data = f.read(packet_length)
            if len(packet_data) < packet_length:
                print("数据包长度不符，文件可能损坏")
                break

            # 在这里处理packet_data，或者将其保存到文件
            print(f"解析到第 {packet_count + 1} 个数据包，长度为 {packet_length} 字节")

            # 示例：将数据包保存到文件
            data = np.ndarray((packet_length, ), np.uint8, buffer=packet_data)
            np.savetxt(f"./tmp/packet_{packet_count}.txt", data)

            packet_count += 1

if __name__ == "__main__":
    # 将此路径替换为实际的packets.bin路径
    file_path = '/home/liuyang/Documents/haisi/my_samples/output/packets.bin'
    parse_packets(file_path)
