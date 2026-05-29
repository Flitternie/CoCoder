import os
import sys

# 获取当前文件的目录（项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将根目录加入系统路径
sys.path.insert(0, current_dir)
