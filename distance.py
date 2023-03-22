import numpy as np
import collections
import heapq
import tifffile
import pickle
import cc3d
import math
import os
from structure_tensor import eig_special_3d, structure_tensor_3d

# unique priority queue
class PrioritySet(object):
    def __init__(self):
        self.heap = []
        self.set = set([])

    def push(self, pri, pair):
        if pair not in self.set:
            heapq.heappush(self.heap, (pri, pair))
            self.set.add(pair)

    def pop(self):
        pri, pair = heapq.heappop(self.heap)
        self.set.remove(pair)
        return pri,list(pair)

    def size(self):
        return len(self.set)

class Supervoxel:
    allvoxels = None
    shape = None
    rawimage = None
    featuremap = None
    orientation = None
    def __init__(self, label, coods, centroid, boundingbox, nbrs, boundary):
        self.id = label  # init stage let label == id. Supervoxel ids <= total labels.
        self.label = label
        self.coods = coods  # should not be changed
        self.centroid = [centroid[0].astype(int),centroid[1].astype(int),centroid[2].astype(int)]
        self.boundingbox = boundingbox
        self.contain = set([self])  # contains other supervoxel's id, this means that they belong to the same supervoxel
        self.nbrs = set(nbrs)  # should not be changed
        self.mother = self
        self.boundary = boundary  # always updated after merge
        self.size = len(coods) #boundary is always not added
    @classmethod
    def writeIntermediateV(cls,labeltowrite,fn):
        '''
        :param labeltowrite: A list of UNIQUE supervoxels we want to write as volume
        :return: Numpy array
        Write Tiff volume which has the coods + boundary of labels
        '''
        v = np.zeros(Supervoxel.shape,dtype=np.int32)
        for l in labeltowrite:
            for idx in Supervoxel.allvoxels[l].getCoods():
                v[idx] = Supervoxel.allvoxels[l].label
            for idx in Supervoxel.allvoxels[l].boundary:
                v[idx] = Supervoxel.allvoxels[l].label
        tifffile.imwrite(fn,v,photometric='minisblack')

    @classmethod
    def writeAll(cls, onlyvisited = None, fna='./intermediate.tif'):
        '''
        :return: Numpy array
        Write merged volume which has the coods + boundary of labels
        '''
        v = np.zeros(Supervoxel.shape,dtype=np.int32)
        seen = set([])
        if onlyvisited != None:
            for i in onlyvisited:
                l = Supervoxel.allvoxels[i]
                if l.label not in seen:
                    #add coods
                    for idx in Supervoxel.allvoxels[l.label].getCoods():
                        v[idx] = l.label
                    #add boundary
                    for idx in Supervoxel.allvoxels[l.label].boundary:
                        v[idx] = l.label
                    seen.add(l.label)

        else:
            for key,voxel in Supervoxel.allvoxels.items():
                if voxel.label not in seen:
                    #add coods
                    for idx in Supervoxel.allvoxels[voxel.label].getCoods():
                        v[idx] = voxel.label
                    #add boundary
                    for idx in Supervoxel.allvoxels[voxel.label].boundary:
                        v[idx] = voxel.label
                    seen.add(voxel.label)

        tifffile.imwrite(fna,v,photometric='minisblack')

    def merge(self, v2):
        '''
        Does my merge allow merging any children voxels?
        :param v2: supervoxel
        :return: Merged supervoxel's IDs exclude mother's id

        Merge 2 supervoxels to the same label. Update centroids, boundaries, labels. Return a list of ids exclude itself.
        '''
        if self.label == v2.label: return None
        bd1, bd2 = self.boundary, v2.boundary
        if self.label < v2.label:
            v2.updateContains(self.mother)  # Update self.monther's contain list, updates all supervoxels's mothers in v2
            self.updateBoundary(bd2)  # Update self.monther's BD in all contain list

        elif self.label > v2.label:
            self.updateContains(v2.mother)  # Update v2.monther's contain list, updates all v1's voxels' mother as v2
            v2.updateBoundary(bd1)  # Update v2's BDs in all contain list

        self.updateCentroid()
        return self.mother.getAllids()

    def getAllids(self):
        '''

        :return: Return a list of ids which belongs to the same supervoxel
        :rtype: list

        Return the list of ids which belongs to the same supervoxel, except itself.
        Always needed to be called on a MONTHER supervoxel
        '''
        temp = set([])
        for obj in self.mother.contain:
            if obj.id == self.id: continue
            temp.add(obj.id)
        return temp

    def getAdjList(self):
        '''

        :return: Return adjacent list (unique labels) of the current supervoxel.
        '''
        temp = set([])
        for obj in self.mother.contain:
            temp.update(obj.uniqueNBR())

        # remove itself from Adj list.
        return temp - set([Supervoxel.allvoxels[self.label]])

    def uniqueNBR(self):
        '''
        :return: List of neighbors with unique labels of one supervoxel.

        Helper function for getAdjList(self)
        '''
        temp2 = set([])
        for l in self.nbrs:
            temp2.add(Supervoxel.allvoxels[l].mother)
        return temp2

    def updateBoundary(self,bd):
        '''
        :param bd: boundary list
        :return: None

        1. Update all boundary pixels in self mother's contain list which includes itself
        2. Must be called after updateContains()
        3. Also add intersection pixels to self.coods
        '''
        intersected = self.boundary.intersection(bd)
        temp = self.boundary.union(bd) - intersected
        # At this point the contain list must be updated in self
        # Must be called after updateContains()

        # add intersection pixels to self.coods
        self.mother.coods.update(intersected)
        voxelsz = len(self.mother.getCoods()) + len(temp)
        for ob in self.mother.contain:
            ob.boundary = temp
            ob.size = voxelsz

    def updateContains(self, newmother):
        '''

        :return: None

        Helper function for merge(),
        1. Update all the label's in the current contain list to the newmother's label.
        2. Update all mothers in the current contain list to the newmother.
        3. Appends all objects in curr mother's contain list to the new mother's contain list
        '''
        newlabel = newmother.label
        for obj in self.mother.contain:
            obj.label = newlabel
            obj.mother = newmother
            newmother.contain.add(obj)



    def updateCentroid(self):
        '''

        :return: None

        1.Update all objects in the mother's contain list which includes itself
        '''
        temp = np.array(list(self.getCoods()))
        if len(temp) < 10:
            print(self)
        new = [np.median(temp[:,0]).astype(int), np.median(temp[:,1]).astype(int), np.median(temp[:,2]).astype(int)]
        for ob in self.mother.contain:
            ob.centroid = new

    # def Get Bounding Box
    def getCoods(self):
        '''

        :return:
        1. Get all coordinates using self.monther
        '''
        temp = set([])
        for ob in self.mother.contain:
            temp.update(ob.coods)
        return temp  # Need to check dimension here

    # def __getattr__(self, name: str):
    #     return self.__dict__[f"_{name}"]

    # def __setattr__(self, name: str, value):
    #     self.__dict__[f"_{name}"] = value

    def __str__(self):
        return f"ID: {self.id}, label: {self.label}, MotherID: {self.mother.id}."


# try 3 metric,
# 1. total intensity / voxel count (V) with dilation scales
# 2. max intensity / voxel count <- NOT A GOOD APPROACH. (x)
# 3. ? SA ?
#
def boundaryIntensity(bdpixels,featuremap,dim=3, mode='avg'):
    '''

    :param bd1: supervoxel 1's boundary pixels
    :param bd2: supervoxel 2's boundary pixels
    :param featuremap: raw image or feature maps
    :param dim: hbrhood size.
    :param featuremap: raw image or feature maps
    :return: return the boundary intensities mark
    '''

    totalIntensity = 0 # for metric 1
    voxelcounts = 0 # for metric 226 ne
    for idx in bdpixels:
        totalIntensity += featuremap[idx]
        voxelcounts+=1
        #iterate neighbours
        if dim > 0:
            for d in range(dim):
                for dx in [-1, 1]:
                    c = list(idx)
                    c[d] += dx
                    if c[d] < 0 or c[d] >= volume.shape[d]: continue
                    intensity = featuremap[tuple(c)]
                    # metric 1
                    totalIntensity+=intensity
                    voxelcounts+=1
    return (totalIntensity/voxelcounts)

#return a normalized surfacenormal weight. bounded between [0,1]
def surfaceNormal(bdpixels, ori1, ori2, pairchosen = 20):
    '''

    :param bdpixels: set contains boundary pixels of supervoxels
    :param ori1: normalized vector for myocyte's orientation at the centroid
    :param ori2: normalized vector for another myocyte's orientation at the centroid
    :param pairchosen: avg number of pairs for metric
    :return: a score
    '''
    scorefrom1 = 0
    scorefrom2 = 0
    k = 0
    while pairchosen>k and bdpixels:
        p1 = bdpixels.pop()
        p2 = bdpixels.pop()
        p3 = bdpixels.pop()
        vector1 = np.array(p1) - np.array(p2)
        vector2 = np.array(p3) - np.array(p2)
        norm = np.cross(vector1,vector2)
        norm = norm / (norm**2).sum()**0.5 #normalize
        scorefrom1 += abs(np.dot(norm,ori1))
        scorefrom2 += abs(np.dot(norm,ori2))
        k+=1

    score=(0.5*scorefrom1+0.5*scorefrom2)/pairchosen
    #print(scorefrom1,scorefrom2)
    return score


def manhattan(x, y, a=20, b=1, c=10):
    '''
    :param x: 3d coordinate of vector 1
    :param y: 3d coordinate of vector 2
    :param a: a * |x1-x2|
    :param b: b * |y1-y2|
    :param c: c * |z1-z2|
    :return: a|x1-x2| + b|y1-y2| + c|z1-z2|

    Computes the manhattan distance of 2 vectors
    '''

    z1, y1, x1 = x
    z2, y2, x2 = y
    return a * abs(x1 - x2)**2 + b * abs(y1 - y2) + c * abs(z1 - z2)


def priorityQueueMerge(L,SAthres=100,totalstep=4000,saveiteration=0):
    '''

    :param iter: iteration to stop
    :param SAthres: contacting area threshold
    :param maxmyocyte: maximum allowed merged size
    :return:
    '''
    visited = collections.defaultdict(bool) #false default
    for k in L:
        visited[k] = False

    #initialize min heap
    minqueue = PrioritySet()
    uniqueIds = set([])
    for vid in visited.keys():
        for nbr in L[vid].getAdjList():  # Here we can either uses smallest edge weight, or PriorityQueue()
            if (vid > nbr.id): continue
            bdpixels = L[vid].boundary.intersection(nbr.boundary)
            if len(bdpixels) < SAthres: continue
            edwt = boundaryIntensity(bdpixels, Supervoxel.featuremap)
            minqueue.push(edwt,frozenset((L[vid],nbr)))

    #Parameters for tuning:
    #totalstep = 4000
    saveiteration = 500
    maxmyocyte = 250000 # 125000 or 250000
    kk = 0
    normalThresh = 0.3
    #SAthres
    while minqueue.size()>0:
        if kk > totalstep:
            break
        edwt,pair = minqueue.pop()
        v1,v2 = pair[0],pair[1] #unpack supervoxel
        if visited[v1.id] or visited[v2.id]: continue #continuemayebe don't need to check label
        if (v1.size+v2.size) > maxmyocyte: continue
        #3000 0.59, 0.58
        if (v1.size+v2.size) > 120000:  #constraint check
            #if v2.size<3000 or v1.size<3000: break
            if edwt > 0.55: continue
            bdpixels = v1.boundary.intersection(v2.boundary)
            ori1 = np.flip(Supervoxel.orientation[:,v1.centroid[0],v1.centroid[1],v1.centroid[2]])
            ori2 = np.flip(Supervoxel.orientation[:,v2.centroid[0],v2.centroid[1],v2.centroid[2]])

            normwt = surfaceNormal(bdpixels,ori1,ori2)

            if normwt < normalThresh: continue


        # MODIFY: We can also check some condition here before merge
        ids = v1.merge(v2)
        #mark visited
        for i in ids:
            visited[i] = True
        #Add all nbrs of the new merged supervoxel to minqueue.
        newvoxel = L[v1.label]
        uniqueIds.add(v1.label)
        for nbr in newvoxel.getAdjList():
            if visited[nbr.id] or visited[nbr.label]: continue
            bdpixels = newvoxel.boundary.intersection(nbr.boundary)
            if len(bdpixels) < SAthres: continue
            edwt = boundaryIntensity(bdpixels, Supervoxel.featuremap)
            minqueue.push(edwt,frozenset((newvoxel,nbr)))

        kk+=1
        # if kk == 3000:
        #     print(132,186)
        #     ori1 = np.flip(Supervoxel.orientation[:,22,75,204])
        #     ori2 = np.flip(Supervoxel.orientation[:,22, 147, 175])
        #     surfaceNormal(L[132].boundary.intersection(L[186].boundary),ori1,ori2)
        if saveiteration and ((kk % saveiteration) == 0 ):
            # save merged turbopixel at each iteration
            #Supervoxel.writeIntermediateV(uniqueIds, f'{kk}.tif')
            uniqueIds = set([]) #update list

    print(kk)
    #uniqueIds
    # Revisit those that are not visited, assign it to the nearest neighbours with the largest CA
    for k,v in visited.items():
        if not visited[k] and L[k].size <= 10000:
            #merge to the neighbour with the largest CA:
            maxBD = -float('inf')
            maxBDnbr = None
            minwt = float('inf')
            minwtnbr = None
            for nbr in L[k].getAdjList():
                bdpixels = L[k].boundary.intersection(nbr.boundary)
                maxBD = max(maxBD,len(bdpixels))
                wt = boundaryIntensity(bdpixels,Supervoxel.featuremap)
                minwt = min(minwt, wt)

                if maxBD == len(bdpixels):
                    maxBDnbr = nbr
                if minwt == wt:
                    minwtnbr = nbr

            if maxBDnbr == minwtnbr and minwtnbr!=None:
                minwtnbr.merge(L[k])
            elif 20 < maxBD and L[k].size<=5000:
                maxBDnbr.merge(L[k])
            elif minwt < 0.6:
                minwtnbr.merge(L[k])
            else:
                continue #ignore this segment.
            visited[k] = True


def mergingCachedBufferStage1(L):
    '''
    :param bsz: 500 Size of cached buffer pairs
    :return:
    '''

    SAthres = 60
    visited = collections.defaultdict(bool) #false default
    for k in L:
        visited[k] = False

    #initialize min heap
    minqueue = PrioritySet()
    uniqueIds = set([])
    for vid in visited.keys():
        for nbr in L[vid].getAdjList():  # Here we can either uses smallest edge weight, or PriorityQueue()
            if (vid > nbr.id): continue
            bdpixels = L[vid].boundary.intersection(nbr.boundary)
            if len(bdpixels) < SAthres: continue
            edwt = boundaryIntensity(bdpixels, Supervoxel.featuremap)
            if edwt>0.5: continue
            minqueue.push(edwt,frozenset((L[vid],nbr)))

    #Parameters for tuning:
    #totalstep = 4000
    bsz = 1000
    maxmyocyte = 250000 # 125000 or 250000
    kk = 0
    normalThresh = 0.3
    voxelchanged = 0
    buffer = PrioritySet() # bufferOneround.size() #add a method in priortiy size which takes another set as input
    #Stage: 1 group all supervoxels as a group of 2
    # Check what happends when not using surface area and edge weight.
    while minqueue.size()>0:
        edwt,pair = minqueue.pop() #visit one voxel each time
        v1,v2 = pair[0],pair[1] #unpack supervoxel
        #Repeat Stage 1 merging, now replace unvisited turbo with new grown units
        if kk==1:
            if (v1.label == 3242 and v2.label == 3922) or (v1.label == 3922 and v2.label == 3242):
                print('error ')


        if minqueue.size() == 0:
            kk += 1  #flag to count
            # Group the rest/unvisited supervoxels based on largest SA among its nbrs.//smallest orientation
            # Should still prevent the edge weight etc.
            unvisitedids = [vid for vid in visited.keys() if not visited[vid]]
            for vid in unvisitedids:
                maxct = 0
                maxNbr = None
                for nbr in L[vid].getAdjList():  # Here we can either uses smallest edge weight, or PriorityQueue()
                    bdpixels = L[vid].boundary.intersection(nbr.boundary)
                    edwt = boundaryIntensity(bdpixels,Supervoxel.featuremap)
                    if edwt>0.5: continue
                    maxct = max(maxct, len(bdpixels))
                    if maxct == len(bdpixels):
                        maxNbr = nbr
                if maxNbr:
                    voxelchanged += 1
                    L[vid].merge(maxNbr)
                    visited[vid] = True
                    uniqueIds.add(L[vid].label) # to be considered in the next round

            print(f'{voxelchanged} voxels have changed at: {kk}, below is unvisited:')
            print([vid for vid in visited.keys() if not visited[vid]])
            #Supervoxel.writeIntermediateV(uniqueIds, f'analysis/SA_newAllvisit{kk}.tif')

            if kk==2: break
            #Push to minqueue:
            SAthres = 2*SAthres
            for vid in uniqueIds:
                for nbr in L[vid].getAdjList():
                    if (vid > nbr.id): continue
                    bdpixels = L[vid].boundary.intersection(nbr.boundary)
                    if len(bdpixels) < SAthres: continue
                    edwt = boundaryIntensity(bdpixels, Supervoxel.featuremap)
                    if edwt>0.4: continue
                    minqueue.push(edwt, frozenset((L[vid].mother, nbr)))
                visited[L[vid].label] = False
            uniqueIds = set([])
            voxelchanged = 0
            continue
        if visited[v1.id] or visited[v2.id]: continue  #continuemayebe don't need to check label

        #if edwt > 0.55: continue
        ids = v1.merge(v2)
        voxelchanged+=1

        # mark mother as visited
        visited[v1.label] = True
        for i in ids:
            visited[i] = True

        newvoxel = Supervoxel.allvoxels[v1.label]
        uniqueIds.add(v1.label)

        # #Add candiadtes to buffer
        # for nbr in newvoxel.getAdjList():
        #     #if visited[nbr.id] or visited[nbr.label]: continue
        #     bdpixels = newvoxel.boundary.intersection(nbr.boundary)
        #     #if len(bdpixels) < SAthres: continue
        #     edwt = boundaryIntensity(bdpixels, Supervoxel.featuremap)
        #     #buffer.push(edwt,frozenset((newvoxel,nbr)))


        # if buffer.size()%bsz==0 or voxelchanged==100:
        #     #add the buffer contents to minqueue.
        #     print(voxelchanged)
        #     Supervoxel.writeIntermediateV(uniqueIds,'analysis/buffer.tif')

    return uniqueIds

#Complete merging in Stage 2
# Favor orientation?
#
#
#Take orientations into account
#non-iterative merging until nothing added to minqueue
#To improve:
#ADD A SURFACE NORMAL METHOD WHICH TAKES 2 SUPERVOXELS DIRECTLY!
#Use saved staged1 tiff as a starting point, speeding things up!
#Cached buffer can be used for check unvisited turbovoxels at a stage
def mergingCachedBufferStage2(uniqueIds, L):
    visited = dict()
    minqueue = PrioritySet()
    for k in L:
        visited[k] = True

    #reinitialze queue
    for vid in uniqueIds:
        visited[vid] = False #add to unvisited
        #Construct minqueue based on intensity stains
        for nbr in L[vid].getAdjList():
            if (vid>nbr.label): continue
            bdpixels = L[vid].boundary.intersection(nbr.boundary)
            if len(bdpixels) < 300: continue
            edwt = boundaryIntensity(bdpixels,Supervoxel.featuremap)
            if edwt > 0.4: continue
            ori1 = np.flip(Supervoxel.orientation[:, L[vid].centroid[0], L[vid].centroid[1], L[vid].centroid[2]])
            ori2 = np.flip(Supervoxel.orientation[:, nbr.centroid[0], nbr.centroid[1], nbr.centroid[2]])
            normwt = surfaceNormal(bdpixels, ori1, ori2)

            minqueue.push(edwt,frozenset((L[vid].mother,nbr)))

    #while queue is non-empty:
    voxelchange = 0
    while minqueue.size()>0:
        normwt,pair = minqueue.pop()
        v1,v2 = pair[0],pair[1]
        if v1.label == v2.label: continue
        #if visited[v1.id] or visited[v2.id]: continue

        ids = v1.merge(v2)
        for i in ids:
            visited[i] = True

        newvoxel = L[v1.label]
        #uniqueIds.add(v1.label)
        #consider merged supervoxels with it's neighbours
        for nbr in newvoxel.getAdjList():
            if visited[nbr.id] or visited[nbr.label]: continue
            bdpixels = newvoxel.boundary.intersection(nbr.boundary)
            #check surface area
            if len(bdpixels) < 300: continue
            #check edgeweight constraint
            edwt = boundaryIntensity(bdpixels,Supervoxel.featuremap)
            if edwt > 0.4: continue
            # calculate normalsurface weight
            ori1 = np.flip(Supervoxel.orientation[:, newvoxel.centroid[0], newvoxel.centroid[1], newvoxel.centroid[2]])
            ori2 = np.flip(Supervoxel.orientation[:, nbr.centroid[0], nbr.centroid[1], nbr.centroid[2]])
            normwt = surfaceNormal(bdpixels, ori1, ori2)
            minqueue.push(edwt,frozenset((newvoxel.mother,nbr)))
        voxelchange += 1

    print(voxelchange)
    Supervoxel.writeIntermediateV(uniqueIds,'./analysis/stage2_edwttight04.tif')


def merging(L, iter, threshold, intermediate=False):
    '''
    :param L: Superpixel list (dictionary?)
    :param iter: Merging iterations
    :param threshold: distance threshold
    :param intermediate: Save supervoxels at each iter
    :param maxmyocyte: Merge only if size < maximum allowed size
    :return: Return A list of included ids after merging
    :rtype: list

    Merge supervoxels based on the threshold, iterations. Return A list of included ids after merging
    '''
    visited = collections.defaultdict(bool) #false default
    uniqueIds = None
    for k in L:
        visited[k] = False
    hasChange = True #flag to tell if supervoxel number changes
    kk = 0
    while iter>kk and hasChange:
        # if mergedIDs:
        #     while mergedIDs: #remove merged supervoxel's
        #         visited[mergedIDs.pop()] = True
        topoL = []
        for vid in visited.keys():
            if visited[vid]: continue
            for nbr in L[vid].getAdjList(): #Here we can either uses smallest edge weight, or PriorityQueue()
                if nbr.id < vid: continue # avoid adding repeated edge
                if visited[nbr.id] or visited[nbr.label]: continue
                bdpixels = L[vid].boundary.intersection(nbr.boundary)
                if len(bdpixels)<threshold: continue
                edgeweight = boundaryIntensity(bdpixels,Supervoxel.featuremap,dim=0)  # manhattan(L[vid].centroid, nbr.centroid) # TWEAK replace with other metric
                minedgeweight = min(edgeweight,minedgeweight) #or TWEAK: take the smallest centroid or closet centroid + inten
                if edgeweight == minedgeweight: #OR manhattan distance threshold
                    minNBR = nbr
            if minNBR:
                topoL.append([L[vid],nbr,minedgeweight])
        # Topological sort based on weights on topoL
        topoL = sorted(topoL, key = lambda x:x[2]) #should only has unvisited supervoxels
        idschanged = set([]) #all ids has been changed
        uniqueIds = set([]) #only monther's label added
        for element in topoL:
            v1,v2,edgeweight = element
            #if edgeweight>threshold: continue
            #make sure each iteration only merges 2
            if v1.label in uniqueIds or v2.label in uniqueIds or v1.label==v2.label: continue
            if not visited[v1.id] or not visited[v2.id]:
                ids = v1.merge(v2)
                idschanged.update(ids)
                uniqueIds.add(v1.label)
                # Updatevisited
                for ii in ids:
                    visited[ii] = True
        kk+=1
        hasChange = len(idschanged)

        if intermediate: #save merged turbopixel at each iteration
            Supervoxel.writeIntermediateV(uniqueIds,f'{kk}.tif')
    #Adds unique ids to visited
    for i in uniqueIds:
        visited[i] = True
    return visited



def updateVolume(visited,voxellist,oldvolume,onlyvisited = False):
    '''

    :param visited: supervoxel' ids that are visited (True) or not visited (False)
    :param voxellist: the dict which contains all supervoxels
    :param oldvolume: Supervoxels' old volume
    :return: New volume after merging

    This function returns the new volume after merging, if onlyvisited=True, we only include visited supervoxels in the new volume.

    '''
    newvolume = np.zeros(oldvolume.shape,dtype=oldvolume.dtype)
    for i in np.ndindex(oldvolume.shape[:3]):
        label = oldvolume[i]
        if label == 0: continue
        if onlyvisited:
            if not visited[label]: continue
        try:
            newvolume[i] = voxellist[label].label
        except:
            newvolume[i] = label

    return newvolume

def writelabel(volume,labeltowrite):
    if isinstance(labeltowrite,int):
        tifffile.imwrite(f'/Users/yananw/Desktop/{labeltowrite}.tif',(volume==labeltowrite),photometric='minisblack')
    else:
        v = np.zeros(volume.shape,dtype=volume.dtype)
        for i in np.ndindex(volume.shape[:3]):
            if volume[i] in labeltowrite:
                v[i] = volume[i]
        tifffile.imwrite('/Users/yananw/Desktop/analysis.tif', v, photometric='minisblack')

def voxelcoordinates(volume):
    '''

    :param volume: input volume
    :param getBD: True, if we want boundary pixels..
    :return: 2 dicts which has 1. coords, 2. boundary

    Given a supervoxel volume, compute each id's coordinates and boundary coordinates
    '''
    coordinates = collections.defaultdict(set)
    boundaries = collections.defaultdict(set)
    def checkNBRs(idx):
        for dim in range(3):
            for dx in [-1,1]:
                c = list(idx)
                c[dim] += dx
                if c[dim]<0 or c[dim]>=volume.shape[dim]: continue
                l = volume[tuple(c)]
                if l>0:
                    boundaries[l].add(idx)


    for i in np.ndindex(volume.shape[:3]):
        label = volume[i]
        if label > 0:
            coordinates[label].add(i)
        else:
            checkNBRs(i)
    return coordinates,boundaries

def saveObj(o,fn,protocal=False):
    if not protocal:
        fs = open(fn,'wb')
        pickle.dump(o,fs)
    else:
        with open(fn,'wb') as fff:
            pickle.dump(o,fff,pickle.HIGHEST_PROTOCOL)

def loadObj(fn):
    fs = open(fn,'rb')
    return pickle.load(fs)

def nbrhood(dim=3):
    return np.full((3, 3, 3), 1.0/(dim**3),dtype=np.float64)



def writeOrientation(img, sigma=3, rho = 80, gap = 10, fn = './orien.tif', write=False):
    '''
    :param img: raw or feature map for structure tensor calculation
    :param sigma: noise level
    :param rho: nbrhood size
    :param gap: discretized step sizes.
    :return: None

    Write structure tensor's largest eigenvector as new Tiff image.
    '''
    #structure tensor
    S = structure_tensor_3d(img.astype(np.float64), sigma, rho)
    val, vec = eig_special_3d(S)
    x_limit, y_limit, z_limit = img.shape

    def helper(center, direction, t=gap*2):
        """ Return the coordinates of a line segment drawn with:
             <x,y,z> = <center_x,center_y,center_z> + t * <direction_x, direction_y, direction_z>
            and t varies from (-t // 2, t // 2) and lies within sizelimit (sizelimit_x,sizelimit_y,sizelimit_z)
        Parameters
        ----------
        center:  <center_x,center_y,center_z>, 3d vector, middle of line segment
        direction: <direction_x, direction_y, direction_z>, 3d vector, slope of line
        sizelimit: 3d vector, canvas size limit
        t: line parametrized in t, drawn with (-t // 2, t // 2)
        Returns
        -------
        coods[:,0]: coordinates x component
        coods[:,1]: coordinates y component
        coods[:,2]: coordinates z component
        """
        x_limit, y_limit, z_limit = img.shape
        cooods = []
        for i in range(-t // 2, t // 2, 1):
            x_new, y_new, z_new = np.array(center) + np.array(direction) * i
            x_new, y_new, z_new = round(x_new), round(y_new), round(z_new)
            allowz = (z_new >= 0 and z_new < z_limit)
            if x_new >= 0 and x_new < x_limit and y_new >= 0 and y_new < y_limit and allowz:
                cooods.append([x_new, y_new, z_new])
                if y_new + 1 < y_limit and [x_new, y_new + 1, z_new] not in cooods:
                    cooods.append([x_new, y_new + 1, z_new])
                if y_new - 1 >= 0 and [x_new, y_new - 1, z_new] not in cooods:
                    cooods.append([x_new, y_new - 1, z_new])
                if x_new + 1 < x_limit and [x_new + 1, y_new, z_new] not in cooods:
                    cooods.append([x_new + 1, y_new, z_new])
                if x_new - 1 >= 0 and [x_new - 1, y_new, z_new] not in cooods:
                    cooods.append([x_new - 1, y_new, z_new])
        cooods = np.array(cooods)
        return cooods[:, 0], cooods[:, 1], cooods[:, 2]
    if write:
        for i in range(gap,x_limit,1):
            for j in range(gap,y_limit,2*gap):
                for k in range(gap, z_limit, 2*gap):
                    print(i,j,k)
                    x,y,z = vec[:,i,j,k]
                    img[helper([i,j,k],[z,y,x],10)] = 4000
        tifffile.imwrite(fn,img,photometric='minisblack')

    with open(fn.replace('.tif','.npy'),'wb') as f:
        np.save(f,vec)


if __name__ == "__main__":

    directory = "/Users/yananw/Desktop/supervoxels"
    voxelthres = 5

    for filename in os.listdir(directory):
        if not filename.endswith('031.tif'): continue
        fn = os.path.join(directory,filename)
        print(fn)
        volume = tifffile.imread(fn)
        print(f"Processing file: {fn}")
        stats = cc3d.statistics(volume)  # bounding_boxes, voxel_counts, centroids
        edges = cc3d.region_graph(volume, connectivity=18)

        pickesuffix = filename.replace('.tif','.pickle')

        if os.path.exists("./coods_"+pickesuffix) and os.path.exists(("./bounds_"+pickesuffix)):
            coordinates = loadObj("./coods_"+pickesuffix)
            boundaries = loadObj("./bounds_"+pickesuffix)
        else:
            coordinates, boundaries = voxelcoordinates(volume)
            saveObj(coordinates,"./coods_"+pickesuffix)
            saveObj(boundaries,"./bounds_"+pickesuffix)
            print(f'Write coordinates to' + "./coods_"+pickesuffix)
            print(f'Write boundaries to' + "./bounds_"+pickesuffix)

        if os.path.exists("./Adj_"+pickesuffix):
            Adj = loadObj("./Adj_"+pickesuffix)
        else:
            # construct adjacent matrix
            Adj = collections.defaultdict(set)
            for e in edges:
                if stats['voxel_counts'][e[0]] < voxelthres or stats['voxel_counts'][e[1]] < voxelthres: continue
                Adj[e[0]].add(e[1])
                Adj[e[1]].add(e[0])

            print(f'Write Adj to' + "./Adj_"+pickesuffix)
            saveObj(Adj,"./Adj_"+pickesuffix)

        N = len(coordinates)

        # create supervoxel
        SupervoxelList = {}
        for i in range(1, N+1):
            if stats['voxel_counts'][i] < voxelthres: continue
            SupervoxelList[i] = Supervoxel(i,
                                           coordinates[i],
                                           stats["centroids"][i],
                                           stats["bounding_boxes"][i],
                                           Adj[i],
                                           boundaries[i])

        Supervoxel.allvoxels = SupervoxelList
        Supervoxel.shape = volume.shape
        Supervoxel.rawimage = tifffile.imread(f'/Users/yananw/Desktop/highResolutionMyocytes/{filename}')
        Supervoxel.featuremap = tifffile.imread(f'/Users/yananw/Desktop/featuremap/{filename}')
        oriFN  = "./" + filename.replace('.tif','')+'ori12.npy'
        #writeOrientation(Supervoxel.rawimage,sigma=2,rho=12,fn=oriFN)
        with open(oriFN,'rb') as ff:
            print(f"Open {oriFN} as orientation file for {filename}")
            Supervoxel.orientation = np.load(ff) #remember it is xyz. need to parse and do zyx!
        #starts merging
        print("Starts Merging...")
        #priorityQueueMerge(SupervoxelList,totalstep=600)
        uniqueids = mergingCachedBufferStage1(SupervoxelList)
        mergingCachedBufferStage2(uniqueids, SupervoxelList)
        #Supervoxel.writeAll(fna=f'./analysis/stage2.tif') #or only visied: ([k for k,v in visited.items() if v])
        # writelabel(volume,[322,118,333,397,399,416])  writelabel(volume, 322)

