import cv2 as cv
import sys
import os
import numpy as np

def get_pixel_color(x, y, img_size_x, img_size_y):
	if (x == img_size_x//2 and y == img_size_y//2):
		return np.array([1, 1, 1])
	center = np.array([img_size_x//2, img_size_y//2], dtype=np.float32)
	p = np.array([x, y], dtype=np.float32) - center
	norm = np.linalg.norm(p)
	vec_p = p / norm
	angle = np.arccos(np.dot(vec_p, np.array([1, 0])))
	if (p[1] < 0):
		angle = np.pi * 2 - angle

	if (angle >= 0 and angle < np.pi/3):
		r = 1
		g = angle / (np.pi/3)
		b = 0
	elif (angle >= np.pi/3 and angle < np.pi*2/3):
		r = 1 - (angle - np.pi/3) / (np.pi/3)
		g = 1
		b = 0
	elif (angle >= np.pi*2/3 and angle < np.pi):
		r = 0
		g = 1
		b = (angle - np.pi*2/3) / (np.pi/3)
	elif (angle >= np.pi and angle < np.pi*4/3):
		r = 0
		g = 1 - (angle - np.pi) / (np.pi/3)
		b = 1
	elif (angle >= np.pi*4/3 and angle < np.pi*5/3):
		r = (angle - np.pi*4/3) / (np.pi/3)
		g = 0
		b = 1
	else:
		r = 1
		g = 0
		b = 1 - (angle - np.pi*5/3) / (np.pi/3)

	return np.array([r, g, b])

def main(point_2d_array_syn, in_syn_im, infile_dir):
    img_size_x = in_syn_im.shape[1]
    img_size_y = in_syn_im.shape[0]
    for i in range(img_size_x):
        for j in range(img_size_y):
            if (point_2d_array_syn[i][j][0] != -1 and point_2d_array_syn[i][j][1] != -1 and point_2d_array_syn[i][j][0] < img_size_x and point_2d_array_syn[i][j][1] < img_size_y):
                cv.circle(in_syn_im, (point_2d_array_syn[i][j][0], point_2d_array_syn[i][j][1]), 1, get_pixel_color(i, j, img_size_x, img_size_y) * 255, thickness=2)
    cv.imshow("Output", in_syn_im)
    cv.waitKey(0)
    cv.imwrite(infile_dir + "/output/output.jpg", in_syn_im)

if __name__ == '__main__':
	if (len(sys.argv) != 2):
		print("Usage: python %s index_of_depth_image" %sys.argv[0])
		print("Please also update the index in top of test_step1 blend file.")
		exit()
	infile_dir = os.getcwd()
	point_2d_array_syn = np.load(infile_dir + '/output/' + '2D_loc_syn.npy')
	in_syn_im = cv.imread(infile_dir + '/depth_' + str(sys.argv[1]) + '.png')
	main(point_2d_array_syn, in_syn_im, infile_dir)
