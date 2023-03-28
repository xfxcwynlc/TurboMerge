import numpy as np
import tifffile
import os
'''
Util functions to generate zero-masks, pre-process turbovoxels in the background. 
'''
def dfs_zeros(img,listcoods):
    '''

    :param img: input image volume
    :param listcoods: list of seeds to place at,[(x1,x2,x3),(y1,y2,y3),...]
    :return: zero masks volume
    '''
    zeromasks = np.ones(img.shape)
    seen = np.zeros(img.shape,dtype='bool')
    x_limit,y_limit,z_limit = img.shape

    def checkboundary(cc1):
        x,y,z = cc1
        if x >= x_limit or x<0:
            return False
        if y>= y_limit or y<0:
            return False
        if z>=z_limit or z<0:
            return False
        return True

    for cood in listcoods:
        stack = [cood]
        while stack:
            cc1 = stack.pop()
            seen[cc1] = True
            zeromasks[cc1] = 0
            top = (cc1[0],cc1[1]-1,cc1[2])
            down =  (cc1[0],cc1[1]+1,cc1[2])
            front = (cc1[0],cc1[1],cc1[2]-1)
            back = (cc1[0],cc1[1],cc1[2]+1)
            for nbr in (top,down,front,back):
                if checkboundary(nbr) and (img[nbr] == 0) and (not seen[nbr]) :
                    stack.append(nbr)
    return zeromasks

def getcoods(x,y,zrange=None):
    '''

    :param x: seed's x location
    :param y: seed's y location
    :param zrange: (start,end) seed appearing from z==start to z==end
    :return:
    '''
    lcoods = []
    for z in range(zrange[0],zrange[1]+1):
        lcoods.append((z,y,x))
    return lcoods

def getallcoods(lcoods,listrgs):
    '''

    :param lcoods: list of seeds, [(x1,y1),(x2,y2),...]
    :param listrgs: list of ranges, [(s1,e1),(s2,e2),...]
    :return: list of all coods which appears as [(x1,y1,s1), (x1,y1,s1+1),..(x1,y1,e1),(x2,y2,s2),..]
    '''
    all = []
    for i in range(len(lcoods)):
        all += getcoods(lcoods[i][0],lcoods[i][1],listrgs[i])
    return all

if __name__ == '__main__':
    # img = tifffile.imread('./018.tif')  # (0,305,22) (0,252,306)
    # listcoods = getallcoods([(129, 199), (251, 233), (197, 31), (318, 0)], [(0, 106), (107, 159), (0, 58), (59, 110)])
    # res = dfs_zeros(img, listcoods)
    # tifffile.imwrite('./018zero.tif', res.astype('uint8'), photometric='minisblack')
    img = tifffile.imread('~/Desktop/supervoxels')
    for i in range(0, 9):
        base = 10
        if i == 4: continue
        raw = tifffile.imread(os.path.join('/Users/yananw/Desktop/supervoxels', '0' + str(base + i) + '.tif'))
        mask = tifffile.imread('0' + str(base + i) + 'zero_new.tif')
        fn = os.path.join('/Users/yananw/Desktop/supervoxels', 'processed_0' + str(base + i) + '.tif')
        tifffile.imwrite(fn, raw * mask, photometric='minisblack')

