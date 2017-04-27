from rtree import index

def transform_noise(x):
    if x >60:
        return 1
    if x >50:
        return 0.6
    else:
        return 0.3

def transform_green(x):
    if x > 0.7:
        return 0.3
    if x > 0.3:
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
    
def categorialize_noise(x):
    if x ==1:
        return 3
    if x ==0.7:
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