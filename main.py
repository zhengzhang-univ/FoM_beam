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
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('there! \n')
