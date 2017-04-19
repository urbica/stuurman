from heapq import heappop, heappush
import geocoder
from numpy import arccos, pi
import geopandas as gp
from itertools import count
from collections import deque
def find_nearest_node(coordinates, index):
    if type(coordinates) == str:
        c = geocoder.yandex(coordinates).latlng
        coordinates = [float(c[x]) for x in [1,0]]
    nearest = tuple(index.nearest(coordinates, 1))
    nearest_node = nearest[0]
    return nearest_node
def get_path(list_of_nodes, dataset):
    data = dataset[(dataset.source.isin(list_of_nodes))&(dataset.target.isin(list_of_nodes))]
    return data.to_json()
def distance(long p1, long p2, dict coords):
    cdef float x1,x2,y1,y2
    x1,y1 = coords[p1]
    x2,y2 = coords[p2]
    return (((x2-x1)**2+(y2-y1)**2)**0.5)*15
def neighs_iter(key, g):
    for x in g[key].items():
        yield x
def bidirectional_astar(G, source_coords, 
                        target_coords, heuristic,
                        spatial_index, dataset, 
                        coords, weight = 'weight'):
    nod = tuple([find_nearest_node(x, spatial_index) for x in [source_coords, target_coords]])
    source,target = nod
    c = [count(), count()]
    queue = [[(0, next(c[0]), source, 0, None)], [(0, next(c[1]), target, 0, None)]]
    enqueued = [{},{}]
    explored = [{}, {}]
    heu = [target,source]
    d=1
    while queue[0] and queue[1]:
        d = 1-d
        _, __, v, dist, parent = heappop(queue[d])
        if v in explored[1-d]:
            #return dl
            path1 = deque([v])
            #path1 = [v]
            path2 = deque([])
            #path2 = []
            node1 = parent
            node2 = explored[1-d][v]
            while node1 is not None:
                #if node1 is not None:
                path1.appendleft(node1)
                node1 = explored[d][node1]
                #if node2 is not None:
                #    path2.append(node2)
                #    node2 = explored[1-d][node2]
            while node2 is not None:
                path2.append(node2)
                node2 = explored[1-d][node2]
            #path1.reverse()
            path1.appendleft(source)
            path2.append(target)
            finalpath = list(path1)+list(path2)
            return get_path(finalpath, dataset)
        
        if v in explored[d]:
            continue

        explored[d][v] = parent
        
        for neighbor, w in neighs_iter(v, G):
            #dl[d].append(neighbor)
            if neighbor in explored[d]:
                continue
            ncost = dist + w.get(weight, 1)
            if neighbor in enqueued[d]:
                qcost, h = enqueued[d][neighbor]
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, heu[d], coords)
            enqueued[d][neighbor] = ncost, h
            heappush(queue[d], (ncost+h, next(c[d]), neighbor, ncost, v))