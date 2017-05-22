#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
上海公积金验证码识别，识别率99%

:date
	2017-05-22
:site
	https://persons.shgjj.com/VerifyImageServlet
"""



from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import numpy as np
import os


WHITE = (255,255,255)
BLACK = (0,0,0)
threshold = 100


clf = joblib.load('../model/fund_sh.m')


def preprocess(im):
	pix = im.load()
	w,h = im.size
	# print w,h
	# print pix[0,0]
	lst = []
	for i in xrange(0,w):
		cnt = 0
		for j in xrange(0,h):
			if sum(pix[i,j])>threshold:
				pix[i,j] = BLACK
				cnt += 1
			else:
				pix[i,j] = WHITE
		if cnt>0:
			lst.append(1)
		else:
			lst.append(0)
	# im.show()
	# print lst
	subims = []
	start = 0
	end = 0
	while(len(subims)<4):
		start = lst.index(1,end)
		try:
			end = lst.index(0,start)
		except:
			end = w
		subim = im.crop((start,0,end,h)).resize((10,16))
		subims.append(subim)
		# subim.show()
	return subims


def im2array(im):
	pix = im.load()
	w,h = im.size
	lst = []
	for i in xrange(0,w):
		for j in xrange(0,h):
			if pix[i,j] == BLACK:
				lst.append(1)
			else:
				lst.append(0)
	return np.array(lst)


def recognize(im):
	result = ''
	for subim in preprocess(im):
		arr = im2array(subim).reshape(1, -1)
		result += clf.predict(arr)[0]
	return result


def test():
	for x in xrange(1,101):
		filename = str(x)+'.jpg'
		im = Image.open('../image/fund_sh/'+filename)
		result = recognize(im)
		print filename + '====>>>>' + result


if __name__ == '__main__':
	test()
