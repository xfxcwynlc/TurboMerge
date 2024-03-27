# TurboMerge
Python implementation of turbovoxel merging 

Class PrioritySet : 
------- 
 Min priority queue with uniqueness
 1. self.heap
 2. self.set
 
Class Supervoxel : 
-----
Abstract data which records turbovoxel's attributes :
1. A mother supervoxel, acts an entity set of all merged supervoxels that belong to this mother class. Init to itself
2. A self contain list
3. Coordinates of the voxel
4. Boundary coordinates of the voxel
5. Centroid of the voxel
6. Orientation of the voxel
7. self ID
8. self label
9. Adjacent neighbours of this object


self methods:
------


class methods:
------
