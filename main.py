from FoM import FoM
import time

t1 = time.time()
# Read beam files
beam_file_path = './'

test = FoM(400, 800, 201, beam_file_path)

figure_of_merit = test.FoM()

t2 = time.time()
print("FoM is {}!\n".format(figure_of_merit))
print("Elapsed time {}".format(t2-t1))

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 ⌘F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('there! \n')

