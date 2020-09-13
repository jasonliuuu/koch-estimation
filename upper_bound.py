import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from numpy.linalg import norm
from operator import mul
from functools import reduce
from itertools import permutations 
# from tqdm import tqdm

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

def diameter(trig):
	if trig.shape[1]==3:
		diff = trig[1][1]-np.array([2/3,sqrt(3)/9])
		_l2 = norm(diff,2)
		return _l2
	else:
		return diameter(trig[0])

def main():
	trig = koch_iter(input_trig=np.array([[0,0],[1/2,sqrt(3)/6],[1,0]]),iteration=3,section=0)
	ret = diameter(trig[1])
	return ret

if __name__ == '__main__':
	print(main())