import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from numpy.linalg import norm
from operator import mul
from functools import reduce
from itertools import permutations 
from tqdm import tqdm

def rotation_matrix(input_vector,center,radians):
    theta = np.radians(-radians)
    c, s = np.cos(theta), np.sin(theta)
    ret = np.zeros_like(input_vector)
    cx, cy = center
    for i in range(input_vector.shape[0]):
        px, py = input_vector[i]
        ret[i] = np.array([
            c*(px-cx)-s*(py-cy)+cx,
            s*(px-cx)+c*(py-cy)+cy])
    return ret

def koch_iter(input_trig=np.array([[0,0],[1/2,sqrt(3)/6],[1,0]]),iteration=4,section=0):
    if iteration ==0:
        return input_trig
    part1 = (input_trig-input_trig[0])/3+input_trig[0]
    part2 = rotation_matrix(part1,part1[0],-60)+(part1[2]-part1[0])
    part3 = rotation_matrix(part1,part1[2],60)+(part1[2]-part1[0])
    part4 = (input_trig-input_trig[2])/3+input_trig[2]
    for i in [part1,part2,part3,part4]:
        for j in i:
            plt.scatter(j[0],j[1],c="purple",marker=".")
            plt.title(f"Fig.3 Plot after {iteration} iterations")
        plt.ylim((0,0.5))
    return np.array([
    	koch_iter(input_trig=part1,iteration=iteration-1,section=1),
    	koch_iter(input_trig=part2,iteration=iteration-1,section=2),
    	koch_iter(input_trig=part3,iteration=iteration-1,section=3),
    	koch_iter(input_trig=part4,iteration=iteration-1,section=4)])

def cal_diameter(arr):
    temp_arr = arr.reshape(reduce(mul,arr.shape[:-1]),2)
    max_val = 0
    all_sets = permutations(temp_arr,2)
    for i in all_sets:
        _l2 = norm(abs(i[1]-i[0]),2)
        max_val = max(max_val,_l2)
    return max_val

def solve(iterations):
    trig = koch_iter(input_trig=np.array([[0,0],[1/2,sqrt(3)/6],[1,0]]),iteration=iterations,section=0)
    trig = trig.reshape(reduce(mul,trig.shape[:-2]),3,2)
    def eval_diameter(k):
        nonlocal trig
        min_val = 1
        for i in range(trig.shape[0]-k):
            min_val = min(cal_diameter(trig[i:i+k]),min_val)
        return min_val
    for k in tqdm(range(1,4**iterations)):
        print(f'i = {iterations}, k = {k}, diameter of the lower bound cover: {eval_diameter(k)}')

def main():
	solve(3)

if __name__ == '__main__':
	main()