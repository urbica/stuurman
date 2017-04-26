from heapq import heappop, heappush
import geocoder
from numpy import arccos, pi
import geopandas as gp
from itertools import count
from collections import deque
from json import dumps

def geocode(text):
    c = geocoder.osm(text).latlng
    coordinates = {'lat':float(c[1]),'lng':float(c[0])}
    return dumps(coordinates)

def find_nearest_node(coordinates, index):
    if type(coordinates) == str:
        c = geocoder.yandex(coordinates).latlng
        coordinates = [float(c[x]) for x in [1,0]]
    nearest = tuple(index.nearest(coordinates, 1))
    nearest_node = nearest[0]
    return nearest_node

def get_path(list_of_nodes, dataset, param = None):
    if param is not None:
        data = dataset[(dataset.source.isin(list_of_nodes))&(dataset.target.isin(list_of_nodes))]
        data = data[['geometry','color_%s'%param]]
        data.rename(columns = {'color_%s'%param:'color'}, inplace=True)
        return data.to_json()
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

def get_w(float w, float g):
    if g is not None:
        return w*g
    else:
        return w

def bidirectional_astar(G, source_coords, 
                        target_coords, heuristic,
                        spatial_index, dataset, 
                        coords, weight = 'weight', additional_param=None):
    
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
            path1 = deque([v])
            path2 = deque([])
            node1 = parent
            node2 = explored[1-d][v]
            while node1 is not None:
                path1.appendleft(node1)
                node1 = explored[d][node1]
            while node2 is not None:
                path2.append(node2)
                node2 = explored[1-d][node2]
            finalpath = list(path1)+list(path2)
            return get_path(finalpath, dataset, additional_param)
        
        if v in explored[d]:
            continue
        
        explored[d][v] = parent
        
        for neighbor, w in neighs_iter(v, G):
            if neighbor in explored[d]:
                continue
            ncost = dist + get_w(w.get(weight, 1), w.get(additional_param,1))
            if neighbor in enqueued[d]:
                qcost, h = enqueued[d][neighbor]
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, heu[d], coords)
            
            enqueued[d][neighbor] = ncost, h
            heappush(queue[d], (ncost+h, next(c[d]), neighbor, ncost, v))

def composite_request(G, source_coords, target_coords, heuristic, spatial_index, dataset, coords):
    green_route =  bidirectional_astar(G, source_coords, target_coords, 
                                       heuristic, spatial_index, dataset, coords, additional_param = 'green')
    poi_route = bidirectional_astar(G, source_coords, target_coords, 
                                       heuristic, spatial_index, dataset, coords, additional_param = 'poi')
    mixed_route = bidirectional_astar(G, source_coords, target_coords, 
                                       heuristic, spatial_index, dataset, coords, additional_param = 'mixed')
    shortest_route = bidirectional_astar(G, source_coords, target_coords, 
                                       heuristic, spatial_index, dataset, coords)
    answer = """[{"id":1,"type":"green","geom":%s},
    {"id":2,"type":"poi","geom":%s},
    {"id":3,"type":"aggregate","geom":%s},
    {"id":4,"type":"shortest","geom":%s}
    ]"""%(green_route,poi_route,mixed_route,shortest_route)
    return answer