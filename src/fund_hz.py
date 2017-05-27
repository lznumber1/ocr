#!/usr/bin/env python
#-*- coding:utf-8 -*-


"""
杭州公积金验证码识别，识别率98%

:date
	2017-05-27
:site
	http://www.hzgjj.gov.cn:8080/WebAccounts/codeMaker

"""



from PIL import Image
from math import sqrt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


WHITE = (255,255,255)
BLACK = (0,0,0)
THRESHOLD = 70

clf = joblib.load('../model/fund_hz.m')



def splity(im):
	"""
	对单个字符的图片在y轴切割
	"""

	w,h = im.size
	pix = im.load()
	s = ''
	for j in xrange(0,h):
		num = 0
		for i in xrange(0,w):
			if pix[i,j] == BLACK:
				num += 1
		if num>0:
			s += '1'
		else:
			s += '0'

	return im.crop((0,s.find('1'),w,s.rfind('1')+1))




def split(im):
	"""
	将图片在x轴方向上切割成4张小图片
	"""

	pix = im.load()
	w,h = im.size
	lst = []
	for i in xrange(0,w):
		num = 0
		for j in xrange(0,h):
			col = pix[i,j]
			if col == BLACK:
				num += 1
		if num > 0:
			lst.append(1)
		else:
			lst.append(0)

	start = 0
	end = 0
	subims = []
	while(start<w and len(subims)<4):
		try:
			start = lst.index(1,start)
			end = lst.index(0,start+1)
		except:
			end = w
		if end-start<5:
			start = end + 1
			continue
		subim = im.crop((start,0,end,h))
		subim = splity(subim)
		subims.append(subim.resize((15,15)))
		start = end + 1
	return subims



def distince(col1,col2):
	"""
	计算两个像素点的欧氏距离
	"""

	return sqrt(pow(col1[0]-col2[0],2)+pow(col1[1]-col2[1],2)+pow(col1[2]-col2[2],2))



def preprocess(im):
	pix = im.load()
	w,h = im.size
	d = {}
	for i in xrange(0,w):
		for j in xrange(0,h):
			col = pix[i,j]
			if d.has_key(col):
				d[col] += 1
			else:
				d[col] = 1
	
	max_col = None
	max_num = 0
	for k,v in d.iteritems():
		# print k,v
		if v > max_num:
			max_col = k
			max_num = v

	# print max_col
	# print '---------------------------------'

	for i in xrange(0,w):
		for j in xrange(0,h):
			dis = distince(pix[i,j],max_col)
			# print dis
			if dis>THRESHOLD:
				pix[i,j] = BLACK
			else:
				pix[i,j] = WHITE

	# im.show()
	return split(im)



def im2list(im):
	"""
	将图片像素点转为0或1属性值
	"""

	w,h = im.size
	pix = im.load()
	lst = []
	for i in xrange(0,w):
		for j in xrange(0,h):
			if pix[i,j] == WHITE:
				lst.append(0)
			else:
				lst.append(1)
	return lst



def recognize(im):
	subims = preprocess(im)
	if len(subims)!=4:
		return None
	pcode = ''
	for subim in subims:
		pcode += clf.predict([im2list(subim)])[0]
	return pcode



def test():
	for i in xrange(1,51):
		filename = str(i) + '.jpg'
		im = Image.open('../image/fund_hz/'+filename)
		pcode = recognize(im)
		print filename+'====>>>>'+pcode



if __name__ == '__main__':
	test()