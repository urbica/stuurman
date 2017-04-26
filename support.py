from rtree import index

def transform_poi(x):
    if x == 3:
        return 1
    if x == 2:
        return 0.5
    if x == 1:
        return 0.7
    if x == 0:
        return 0.5

def transform_green(x):
    if x >0.3:
        return 0.5
    else:
        return 1
    
def categorialize(x):
    if x > 0.7:
        return 3
    if x > 0.3:
        return 2
    else:
        return 1
    
def categorialize_poi(x):
    if x > 0.5:
        return 3
    if x > 0.3:
        return 2
    else:
        return 1
def set_spatial_index(coordinates):
    p = index.Property()
    p.dimension = 2
    ind= index.Index(properties=p)
    for x,y in zip(coordinates.keys(),coordinates.values()):
        ind.add(x,y)
    return ind