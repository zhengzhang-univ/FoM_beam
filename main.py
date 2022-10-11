import mpiutil
from FoM import FoM
import time
import numpy as np

t1 = time.time()
# Read beam files
beam_file_path = '../'

test = FoM(400, 800, 201, beam_file_path)

fom1 = test.FoM()

fom2 = test.FoM_v2()

t2 = time.time()
if mpiutil.rank0:
    print("The FoM (version1) is {}!\n".format(fom1))
    # np.set_printoptions(threshold=np.inf)
    print("The FoM (version2) is {}!\n".format(fom2))
    print("Elapsed time {}".format(t2-t1))


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('there! \n')
