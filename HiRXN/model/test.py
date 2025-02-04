import os

# 获取当前目录
print(os.getcwd())
print(os.path.abspath(os.path.dirname(__file__)))

# # 获取上级目录
print(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
print(os.path.abspath(os.path.dirname(os.getcwd())))
print(os.path.abspath(os.path.join(os.getcwd(), "..")))

# 获取上上级目录
print(os.path.abspath(os.path.join(os.getcwd(), "../..")))

# 获取上上上级目录
print(os.path.abspath(os.path.join(os.getcwd(), "../../..")))

# # 调用上级目录下另一个文件夹内的文件
up_dir_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
demo_path = os.path.join(up_dir_path, "utils/demo.txt")
print(demo_path)
