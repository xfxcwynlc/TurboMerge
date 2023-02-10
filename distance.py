import numpy as np
import collections
import heapq
import tifffile
import pickle
import cc3d
import math

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
    def __init__(self, label, coods, centroid, boundingbox, nbrs, boundary):
        self.id = label  # init stage let label == id. Supervoxel ids <= total labels.
        self.label = label
        self.coods = coods  # should not be changed
        self.centroid = centroid
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
                v[idx] = l
            for idx in Supervoxel.allvoxels[l].boundary:
                v[idx] = l
        tifffile.imwrite(fn,v,photometric='minisblack')

    @classmethod
    def writeAll(cls, onlyvisited = None):
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

        tifffile.imwrite('./intermediatemerged.tif',v,photometric='minisblack')

    def merge(self, v2):
        '''

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
        voxelsz = len(self.mother.coods)
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
        new = np.array([np.median(temp[0, :]), np.median(temp[1, :]), np.median(temp[2, :])])
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
def boundaryIntensity(bd1,bd2,featuremap,dim=3, mode='avg'):
    '''

    :param bd1: supervoxel 1's boundary pixels
    :param bd2: supervoxel 2's boundary pixels
    :param featuremap: raw image or feature maps
    :param dim: hbrhood size.
    :param featuremap: raw image or feature maps
    :return: return the boundary intensities mark
    '''

    bdpixels = bd1.intersection(bd2)
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
    return (totalIntensity/voxelcounts),len(bdpixels)




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


def priorityQueueMerge(L,SAthres=100):
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
            edwt,bdsz = boundaryIntensity(L[vid].boundary, nbr.boundary, Supervoxel.featuremap)
            if bdsz < SAthres : continue
            minqueue.push(edwt,frozenset((L[vid],nbr)))

    #Parameters for tuning:
    totalstep = 3000
    saveiteration = 500
    maxmyocyte = 100000 # 125000 or 250000
    kk = 0
    #SAthres
    while minqueue.size()>0:
        if kk > totalstep:
            break
        edwt,pair = minqueue.pop()
        v1,v2 = pair[0],pair[1] #unpack supervoxel
        if visited[v1.id] or visited[v2.id]: continue #continuemayebe don't need to check label
        if (v1.size+v2.size) > maxmyocyte: continue

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
            edwt,bdsz = boundaryIntensity(newvoxel.boundary, nbr.boundary, Supervoxel.featuremap)
            if bdsz < SAthres: continue
            minqueue.push(edwt,frozenset((newvoxel,nbr)))

        kk+=1
        if saveiteration and ((kk % saveiteration) == 0 ):
            # save merged turbopixel at each iteration
            Supervoxel.writeIntermediateV(uniqueIds, f'{kk}.tif')
            uniqueIds = set([]) #update list

    print(kk)


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
                edgeweight,bdsz = boundaryIntensity(L[vid].boundary, nbr.boundary,Supervoxel.featuremap,dim=0)  # manhattan(L[vid].centroid, nbr.centroid) # TWEAK replace with other metric
                if edgeweight>threshold: continue

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
                if 2037 in ids:
                    print(ids)
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
        if i == (8,30,1):
            print(i,label)

        if label > 0:
            coordinates[label].add(i)
        else:
            checkNBRs(i)
    return coordinates,boundaries

def saveObj(o,fn):
    fs = open(fn,'wb')
    pickle.dump(o,fs)

def loadObj(fn):
    fs = open(fn,'rb')
    return pickle.load(fs)

def nbrhood(dim=3):
    return np.full((3, 3, 3), 1.0/(dim**3),dtype=np.float64)

if __name__ == "__main__":
    # supervoxel img
    img = tifffile.imread("/Users/yananw/Desktop/supervoxels/031.tif")
    volume = img #[0:30, :, :]
    voxelthres = 3

    stats = cc3d.statistics(volume)  # bounding_boxes, voxel_counts, centroids
    edges = cc3d.region_graph(volume, connectivity=18)
    # construct adjacent matrix
    Adj = collections.defaultdict(set)
    for e in edges:
        if stats['voxel_counts'][e[0]] < voxelthres or stats['voxel_counts'][e[1]]< voxelthres:
            continue
        Adj[e[0]].add(e[1])
        Adj[e[1]].add(e[0])

    #coordinates,boundaries = voxelcoordinates(img)
    #store it as pickle object
    #saveObj(coordinates,'./31coords.pickle')
    #saveObj(boundaries,'./31bounds.pickle')

    coordinates = loadObj('./31coords.pickle')
    boundaries = loadObj('./31bounds.pickle')

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

    # Stores SupervoxelList as class attribute:
    Supervoxel.allvoxels = SupervoxelList
    Supervoxel.shape = volume.shape
    #Supervoxel.rawimage = tifffile.imread('/Users/yananw/Desktop/highResolutionMyocytes/19B3s1LV/wga/031.tif')
    Supervoxel.featuremap = tifffile.imread('/Users/yananw/Desktop/featuremap/19B3s1LV/031.tif')

    #starts merging
    priorityQueueMerge(SupervoxelList)
    Supervoxel.writeAll() #or only visied: ([k for k,v in visited.items() if v])
    # writelabel(volume,[322,118,333,397,399,416])  writelabel(volume, 322)

