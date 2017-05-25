#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
大连公积金验证码识别，识别率99%

:date
	2017-05-25
:site
	https://bg.gjj.dl.gov.cn/person/logon/showValidImage.act
"""


from PIL import Image
import numpy as np
from skimage.measure import label
from collections import OrderedDict
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib



WHITE = (255,255,255)
BLACK = (0,0,0)


def splity(im):
	s = ''
	w,h = im.size
	pix = im.load()
	for j in xrange(h):
		num = 0
		for i in xrange(w):
			# print pix[i,j]
			if pix[i,j]==BLACK:
				num += 1
		if num > 0:
			s += '1'
		else:
			s += '0'
	# print s
	start = s.find('1')
	end = s.rfind('1')
	return im.crop((0,start,w,end))


def split(im):	
	w,h = im.size
	pix = im.load()
	lst = []
	for i in xrange(0,w):
		cnt = 0
		for j in xrange(0,h):
			if pix[i,j] == BLACK:
				cnt += 1
		if cnt>0:
			lst.append(1)
		else:
			lst.append(0)

	# print lst

	subims = []
	start = 0
	while start<len(lst):
		if lst[start]==0:
			start = start+1
		else:
			end = lst.index(0,start)
			subim = im.crop((start,0,end,h))
			subim = splity(subim)
			if subim.size[0]>10 or subim.size[1]>10:
				subims.append(subim.resize((30,30)))
			start = end+1

	return subims



def preprocess(im):
	pix = im.load()
	w,h = im.size
	lst = []
	for i in xrange(0,w):
		arr = []
		for j in xrange(0,h):
			if i==0 or j==0 or i==w-1 or j==h-1:
				pix[i,j] = WHITE
				arr.append(0)
			else:
				col = pix[i,j]
				if col != WHITE:
					if (pix[i-1,j]==WHITE and pix[i+1,j]==WHITE) or (pix[i,j-1]==WHITE and pix[i,j+1]==WHITE):
						pix[i,j] = WHITE
						arr.append(0)
					else:
						pix[i,j] = BLACK
						arr.append(1)
				else:
					arr.append(0)

		lst.append(arr)

	
	array = np.array(lst)
	lbarr = label(array,connectivity = 1)
	cnts = OrderedDict()
	for i in lbarr:
		for j in i:
			if j==0:
				continue
			if cnts.has_key(j):
				cnts[j] += 1
			else:
				cnts[j] = 1
	# print cnts

	top_cnts = []
	for k,v in cnts.iteritems():
		if v>10:
			top_cnts.append(k)
	# print top_cnts

	for i in xrange(0,w):
		for j in xrange(0,h):
			if(lbarr[i,j] in top_cnts):
				pix[i,j] = BLACK
			else:
				pix[i,j] = WHITE


	return split(im)



def im2list(im):
	w,h = im.size
	lst = []
	pix = im.load()
	for i in xrange(w):
		for j in xrange(h):
			if pix[i,j] == BLACK:
				lst.append(1)
			else:
				lst.append(0)
	return lst



def test():
	clf = joblib.load('../model/fund_dl.m')
	for i in xrange(1,101):
		im = Image.open('../image/fund_dl/'+str(i)+'.jpg')
		subimgs = preprocess(im)
		if(len(subimgs)!=4):
			print str(i)+'.jpg====>>>>识别不出来'
			continue
		pcode = ''
		for subimg in subimgs:
			r = clf.predict([im2list(subimg)])
			pcode += r[0]
		print str(i)+'.jpg====>>>>'+pcode



if __name__ == '__main__':
	test()
