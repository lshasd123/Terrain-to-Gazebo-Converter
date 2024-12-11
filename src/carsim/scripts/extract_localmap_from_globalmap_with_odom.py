import rosbag
import cv2
import numpy as np
import os
import math
from glob import glob
def process_elevation_map_to_video(png_path, bag_path, pose_topic, video_output="output.avi", scale_factor=0.6):
    """
    根据rosbag中的odom信息，从全局高程图中提取局部高程图并生成视频
    :param png_path: 全局高程图路径
    :param bag_path: rosbag文件路径
    :param pose_topic: pose话题名称
    :param video_output: 输出视频路径
    :param scale_factor: 缩放系数 (默认0.6)
    """
    # 加载PNG图像
    elevation_map = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if elevation_map is None:
        raise ValueError(f"无法加载PNG文件: {png_path}")
    print(f"PNG图像尺寸: {elevation_map.shape}")

    # PNG每像素对应0.2米，应用缩放系数
    resolution = 0.2 * scale_factor  # 每像素实际尺寸（米）
    print(f"每像素实际尺寸: {resolution} 米")

    # ROSBag加载
    bag = rosbag.Bag(bag_path)
    print(f"打开ROSBag文件: {bag_path}")
    # 定义圆半径
    radius = 200  # 示例半径
    frame_width = int(2*radius/math.sqrt(2))
    # 视频初始化
     # 将截取的局部高程图写入视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_output, fourcc, 60, (frame_width, frame_width), isColor=False)

    # 遍历ROSBag中的odom消息
    for topic, msg, t in bag.read_messages(topics=[pose_topic]):
        # 提取位置和航向角
        x = msg.pose.position.x
        y = -msg.pose.position.y
        orientation = msg.pose.orientation
        _, _, yaw = euler_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)
        yaw -= math.pi / 2  # 调整航向角
        # 将位置转换为图像像素坐标
        center_x = int(x /resolution/ resolution) + elevation_map.shape[0] // 2
        center_y = int(y /resolution/ resolution) + elevation_map.shape[1] // 2

        # 提取圆形区域并旋转
        local_map = extract_circular_region(elevation_map,
                                             center_x, center_y, radius, yaw)
        
        local_map = extract_square_region(local_map, radius, radius, frame_width)
        local_map_noisy = add_gaussian_noise_to_partial_pixels(local_map, 
                                                               mean=0, std=20, 
                                                               noise_ratio=0.02)
       

        frame = cv2.resize(local_map_noisy, (frame_width, frame_width))
        video_writer.write(frame)

    bag.close()
    video_writer.release()
    print(f"视频保存完成: {video_output}")

def extract_square_region(elevation_map, center_x, center_y, side_length):
    """
    提取正方形区域。

    参数:
        elevation_map: 输入的高程图 (numpy array)。
        center_x: 正方形中心的 x 坐标 (像素单位)。
        center_y: 正方形中心的 y 坐标 (像素单位)。
        side_length: 正方形区域的边长 (像素单位)。

    返回:
        提取后的正方形区域 (numpy array)。
    """
    height, width = elevation_map.shape

    # 提取正方形区域的边界框
    x_min = max(0, center_x - side_length // 2)
    x_max = min(width, center_x + side_length // 2)
    y_min = max(0, center_y - side_length // 2)
    y_max = min(height, center_y + side_length // 2)

    square_region = elevation_map[y_min:y_max, x_min:x_max]

    return square_region

def euler_from_quaternion(x, y, z, w):
    """
    四元数转欧拉角
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z
import cv2
import numpy as np
import math

def add_gaussian_noise_to_partial_pixels(image, mean=0, std=10, noise_ratio=0.02):
    """
    在图像的一部分像素上添加高斯噪声
    :param image: 输入图像，NumPy数组
    :param mean: 高斯噪声的均值
    :param std: 高斯噪声的标准差
    :param noise_ratio: 添加噪声的像素比例
    :return: 添加部分噪声后的图像
    """
    # 临时转换为浮点类型
    noisy_image = image.astype(np.float32)
    num_pixels = image.size  # 总像素数量
    height, width = image.shape
    num_noisy_pixels = int(num_pixels * noise_ratio)  # 添加噪声的像素数量

    # 随机选择像素的索引
    y_indices = np.random.randint(0, image.shape[0], num_noisy_pixels)
    x_indices = np.random.randint(0, image.shape[1], num_noisy_pixels)

    # 添加高斯噪声到选定像素
    noise = np.random.normal(mean, std, num_noisy_pixels)  # 生成高斯噪声
    noisy_image[y_indices, x_indices] += noise  # 修改选定像素值

    # 限制像素值范围，并转换回 uint8 类型
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    # 创建一个与图像同大小的掩码，用于标记噪声位置
    noise_mask = np.zeros_like(image, dtype=np.uint8)
    noise_mask[y_indices, x_indices] = 1
    kernel_size = 5
    # 对噪声位置进行高斯平滑
    smoothed_image = noisy_image.copy()
    for y, x in zip(y_indices, x_indices):
        # 提取当前像素的局部邻域
        y_min = max(0, y - kernel_size // 2)
        y_max = min(height, y + kernel_size // 2 + 1)
        x_min = max(0, x - kernel_size // 2)
        x_max = min(width, x + kernel_size // 2 + 1)

        # 提取邻域并应用高斯模糊
        region = noisy_image[y_min:y_max, x_min:x_max]
        region_blurred = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)

        # 替换噪声点值为平滑后的中心值
        smoothed_image[y, x] = region_blurred[kernel_size // 2, kernel_size // 2]

    return smoothed_image


def extract_circular_region(elevation_map, center_x, center_y, radius_px, yaw):
    """
    提取圆形区域并旋转。

    参数:
        elevation_map: 输入的高程图 (numpy array)。
        center_x: 圆心的 x 坐标 (像素单位)。
        center_y: 圆心的 y 坐标 (像素单位)。
        radius_px: 圆形区域的半径 (像素单位)。
        yaw: 旋转角 (以弧度表示)。

    返回:
        提取并旋转后的圆形区域 (numpy array)。
    """
    height, width = elevation_map.shape
      
    # 创建圆形掩码
    y_indices, x_indices = np.ogrid[:height, :width]
    distance = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    circular_mask = distance <= radius_px

    # 应用掩码提取圆形区域
    circular_region = np.zeros_like(elevation_map)
    circular_region[circular_mask] = elevation_map[circular_mask]

    # 提取圆形区域的边界框
    x_min = max(0, center_x - radius_px)
    x_max = min(width, center_x + radius_px)
    y_min = max(0, center_y - radius_px)
    y_max = min(height, center_y + radius_px)

    cropped_region = circular_region[y_min:y_max, x_min:x_max]

    # 调整坐标以适应裁剪后的区域
    cropped_center_x = center_x - x_min
    cropped_center_y = center_y - y_min

    # 创建旋转矩阵绕裁剪区域中心旋转
    M = cv2.getRotationMatrix2D((cropped_center_x, cropped_center_y), -math.degrees(yaw), 1.0)
    rotated_circular_region = cv2.warpAffine(cropped_region, M, (cropped_region.shape[1], cropped_region.shape[0]),
                                             flags=cv2.INTER_LINEAR, borderValue=0)

    # 调整为指定的输出尺寸
    output_size = (radius_px * 2, radius_px * 2)
    resized_region = cv2.resize(rotated_circular_region, output_size)

    return resized_region


def process_all_elevation_maps_to_videos(png_dir, bag_dir, output_dir, pose_topic_name):
    """
    处理指定目录下的所有 PNG 文件，与对应的 BAG 文件生成视频。
    
    参数:
        png_dir: PNG 文件所在目录。
        bag_dir: BAG 文件所在目录。
        output_dir: 输出视频的存放目录。
        pose_topic_name: odom 话题名称。
    """
    # 获取 PNG 文件列表
    png_files = glob(os.path.join(png_dir, "*.png"))
    
    if not png_files:
        print(f"未在目录 {png_dir} 中找到 PNG 文件。")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for png_file in png_files:
        # 获取文件名（不带扩展名）
        base_name = os.path.splitext(os.path.basename(png_file))[0]
        # 构造对应的 BAG 文件路径
        bag_file = os.path.join(bag_dir, f"{base_name}.bag")
        # 构造输出视频路径
        video_output = os.path.join(output_dir, f"{base_name}.avi")
        
        # 检查 BAG 文件是否存在
        if not os.path.exists(bag_file):
            print(f"未找到对应的 BAG 文件: {bag_file}，跳过...")
            continue
        
        print(f"处理文件: {png_file} 和 {bag_file}, 输出到 {video_output}")
        
        # 调用主处理函数
        process_elevation_map_to_video(png_file, bag_file, pose_topic_name, video_output=video_output)



if __name__ == "__main__":
    png_dir = "/home/liji/grid_map_ws/src/grid_map_pcl/data"
    bag_dir = "/home/liji/Terrain-to-Gazebo-Converter/bag"
    output_dir = "/home/liji/Terrain-to-Gazebo-Converter/video"
    pose_topic_name = "/racebot/true_state/center_pose"  # 修改为你的 odom 话题名称

    process_all_elevation_maps_to_videos(png_dir, bag_dir, output_dir, pose_topic_name)