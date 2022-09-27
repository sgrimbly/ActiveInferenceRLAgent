import numpy as np
from numpy import random

def randPos(previous_positions):
    '''Return previous_positions array with last element being a new, novel random position that has not already existed in previous_positions. Positions are limited to integer from 1 to 100, inclusive.'''
    pos = random.randint(1,101) # Sample uniformly from [1,100]
    while pos in previous_positions: # If pos is in previous_positions, choose a new one.
        pos = random.randint(1,101)
    previous_positions.append(pos)
    return previous_positions # Return given list with pos appended.



# Test and debug functions
if __name__ == '__main__':
    x = randPos([1,2,3,4])
    print(x)
