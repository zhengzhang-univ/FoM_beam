from FoM import FoM


# Read beam files
beam_file_path = './beam_data_hirax'

test = FoM(400, 800, 201, "/Users/zheng/Dropbox/CarlaShareSept2022/Data 02_09_2022/")

figure_of_merit = test.FoM()

print("FoM is " + figure_of_merit + ", done!")

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 ⌘F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('there! \n')

