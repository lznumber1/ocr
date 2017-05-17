#!usr/bin/env python
# -*- coding:utf-8 -*-


"""
北京公积金验证码识别，识别率99%

:date
	2017-05-17
:site
	http://www.bjgjj.gov.cn/wsyw/servlet/PicCheckCode1
"""

from PIL import Image,ImageFilter
from collections import OrderedDict
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


WHITE = (255,255,255)
BLACK = (0,0,0)


def preprocess(im):
	""" 
	该方法用于去噪分割图片
	:param im: image对象
	:returns: 切割出的小图片列表
	"""
	col_cnt = OrderedDict()
	w,h = im.size
	pix = im.load()
	# print pix[0,0]
	# print pix[1,1]
	for i in xrange(0,w):
		for j in xrange(0,h):
			rgb = pix[i,j]
			if col_cnt.has_key(rgb):
				col_cnt[rgb] += 1
			else:
				col_cnt[rgb] = 1
	# print col_cnt
	d = OrderedDict()
	for k,v in col_cnt.iteritems():
		if v > 10 and v < 100:
			# print k,v
			d[k] = v
	# print len(d)

	subimgs = []
	for k in d.keys():
		left = 100
		upper = 100
		right = 0
		lower = 0
		for i in xrange(0,w):
			for j in xrange(0,h):
				if pix[i,j] == k:
					if left > i:
						left = i
					if right < i:
						right = i
					if upper > j:
						upper = j
					if lower < j:
						lower = j

		subim = im.crop((left,upper,right+1,lower+1))
		subpix = subim.load()
		sw,sh = subim.size
		for i in xrange(0,sw):
			for j in xrange(0,sh):
				if subpix[i,j] == k:
					subpix[i,j] = BLACK
				else:
					subpix[i,j] = WHITE

		subimgs.append(subim.resize((20,15)))

	return subimgs


def im2lst(im):
	""" 
	将图片转为list对象
	:param im: image对象
	:returns: 图片像素列表
	"""
	w,h = im.size
	lst = []
	pix = im.convert('L').load()
	for i in xrange(w):
		for j in xrange(h):
			lst.append(pix[i,j])
	return lst


def test():
	clf = joblib.load('../model/fund_bj.m')
	for x in xrange(1,101):
		filename = str(x)+'.jpg'
		im = Image.open('../image/fund_bj/'+filename)
		subimgs = preprocess(im)
		pcode = ''
		for subimg in subimgs:
			lst = im2lst(subimg)
			r = clf.predict([lst])
			pcode += r[0]
		print '====>>>> %s:%s' % (filename,pcode)


if __name__ == '__main__':
	test()
