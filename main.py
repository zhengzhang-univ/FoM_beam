import mpiutil
from FoM import FoM
import time

t1 = time.time()
# Read beam files
beam_file_path = '../'

test = FoM(400, 800, 201, beam_file_path)

figure_of_merit_1 = test.FoM(include_1st_order = True)
figure_of_merit = test.FoM(include_1st_order = False)

t2 = time.time()
if mpiutil.rank0:
    print("FoM with the first order term is\n {}!\n".format(figure_of_merit_1))
    print("FoM w/o the first order term is\n {}!\n".format(figure_of_merit))
    print("Elapsed time {}".format(t2-t1))


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('there! \n')
