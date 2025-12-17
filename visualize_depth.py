import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# 设置深度图文件夹路径
# 请确保这里的路径是你存放 .tiff 文件的地方
depth_folder = "output/sea-thru-1/train/ours_30000/vis"

# 修改输出文件夹名以便区分，你可以改回 "colored"
output_folder = os.path.join(depth_folder, "colored") 
os.makedirs(output_folder, exist_ok=True)

print(f"Searching in: {depth_folder}")
tiff_files = glob.glob(os.path.join(depth_folder, "*.tiff"))
print(f"Found {len(tiff_files)} files.")

# 遍历所有 tiff 文件
for tiff_path in tiff_files:
    # 1. 读取浮点深度
    depth = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"Could not read: {tiff_path}")
        continue
    
    # 2. 归一化 (把深度值缩放到 0-1 之间以便上色)
    min_val, max_val = depth.min(), depth.max()
    
    # 增加一个安全检查，防止最大值等于最小值时除以零报错
    if max_val - min_val > 1e-5:
        depth_norm = (depth - min_val) / (max_val - min_val)
    else:
        depth_norm = np.zeros_like(depth) # 如果图像是平的，就全黑

    # 3. 应用颜色映射 (关键修改点)
    # 原来是 plt.cm.turbo
    # 现在改为 plt.cm.viridis_r (反转的 viridis)
    # 效果：数值小(远)为亮黄色，数值大(近)为深蓝紫色
    depth_colored = plt.cm.viridis_r(depth_norm)[:, :, :3] # 去掉 alpha 通道
    
    # 4. 转为 BGR 格式保存
    # 先从 0-1 浮点转为 0-255 整数
    depth_colored = (depth_colored * 255).astype(np.uint8)
    # matplotlib 是 RGB，OpenCV 需要 BGR
    depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    
    # 5. 保存
    save_name = os.path.basename(tiff_path).replace(".tiff", ".png")
    cv2.imwrite(os.path.join(output_folder, save_name), depth_colored_bgr)
    print(f"Saved: {save_name}")

print("Done!")