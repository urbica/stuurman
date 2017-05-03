from heapq import heappop, heappush
import geopandas as gp
from itertools import count
from collections import deque
from json import dumps


def find_nearest_node(coordinates, index):
    nearest = tuple(index.nearest(coordinates, 1))
    nearest_node = nearest[0]
    return nearest_node

def get_path(list_of_edges, dataset, param):
    if param != 'weight':
        data = dataset[dataset.id.isin(list_of_edges)]
        data = data.rename(columns={'color_%s'%param:'color'})
        data = data[['geometry','color']]
        return data.to_json()
    else:
        data = dataset[dataset.id.isin(list_of_edges)]
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
                        coords, additional_param='weight'):
    
    nod = tuple([find_nearest_node(x, spatial_index) for x in [source_coords, target_coords]])
    source,target = nod
    queue = [[(0, source, 0, None, None)], [(0, target, 0, None, None)]]
    enqueued = [{},{}]
    explored = [{}, {}]
    edge_parent = [{}, {}]
    heu = [target,source]
    d=1
    
    while queue[0] and queue[1]:
        d = 1-d
        _, v, dist, parent, edge = heappop(queue[d])
        
        if v in explored[1-d]:
            if v is not None:
                path1 = deque([edge])
                w = G[explored[1-d][v]][v]
                path2 = deque([w.get('id',1)])
            else:
                path1 = deque([edge])
                path2 = deque([])
            node1 = parent
            node2 = explored[1-d][v]
            
            while node1 is not None:
                path1.appendleft(edge_parent[d][node1])
                node1 = explored[d][node1]
                
            while node2 is not None:
                path2.append(edge_parent[1-d][node2])
                node2 = explored[1-d][node2]
                
            finalpath = list(path1)+list(path2)
            return get_path(finalpath, dataset, additional_param)
        
        if v in explored[d]:
            continue
        
        explored[d][v] = parent
        edge_parent[d][v] = edge
        
        for neighbor, w in neighs_iter(v, G):
            if len(G[neighbor])==1:
                continue
            if neighbor in explored[d]:
                continue
            ncost = dist + w.get(additional_param,1)
            if neighbor in enqueued[d]:
                qcost, h = enqueued[d][neighbor]
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, heu[d], coords)
            
            enqueued[d][neighbor] = ncost, h
            e = w.get('id',1)
            heappush(queue[d], (ncost+h, neighbor, ncost, v, e))

            
def composite_request(G, source_coords, target_coords, heuristic, spatial_index, dataset, coords):
    green_route =  bidirectional_astar(G, source_coords, target_coords, 
                                       heuristic, spatial_index, dataset, coords, 'green')
    noisy_route = bidirectional_astar(G, source_coords, target_coords, 
                                       heuristic, spatial_index, dataset, coords, 'noise')
    air_route = bidirectional_astar(G, source_coords, target_coords, 
                                       heuristic, spatial_index, dataset, coords, 'air')
    shortest_route = bidirectional_astar(G, source_coords, target_coords, 
                                       heuristic, spatial_index, dataset, coords)
    answer = """[{"id":1,"type":"green","geom":%s},
    {"id":2,"type":"noise","geom":%s},
    {"id":2,"type":"air","geom":%s},
    {"id":3,"type":"shortest","geom":%s}
    ]"""%(green_route,noisy_route,air_route,shortest_route)
    return answer