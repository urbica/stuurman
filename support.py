from rtree import index
from numpy import percentile

def colorize(column):
    data = []
    p33 = percentile(column.values, 33)
    p66 = percentile(column.values, 66)
    for x in column.values:
        if x > p33:
            data.append(2)
        elif x > p66:
            data.append(3)
        else:
            data.append(1)
    return data
    
def set_spatial_index(coordinates):
    p = index.Property()
    p.dimension = 2
    ind= index.Index(properties=p)
    for x,y in zip(coordinates.keys(),coordinates.values()):
        ind.add(x,y)
    return ind