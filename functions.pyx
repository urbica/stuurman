from math import pow
from heapq import heappop, heappush
from rtree import index
from itertools import count
from collections import deque
from numpy import arccos, pi

def check_similarity(list l,list l2):
    i =  len(set(l2)-set(l))
    i2 =  len(set(l)-set(l2))
    if i/len(l2) > 0.9 or i2/len(l2) > 0.9:
        return True
    return False
    
def vector_dist(tuple vector):
    cdef float d
    d = vector[0]**2+vector[1]**2
    return d**0.5

def get_circ(tuple vector1, tuple vector2):
    cdef float scal, evklid1, evklid2, total_evklid, answer
    scal = vector1[0]*vector2[0]+vector1[1]*vector2[1]
    evklid1,evklid2  = [vector_dist(vect) for vect in (vector1, vector2)]
    total_evklid = evklid1*evklid2
    answer = scal/total_evklid
    return arccos(answer)
    
def get_vector(long node1,long node2, dict list_of_coords):
    cdef tuple coords1, coords2
    cdef float x, y
    coords1 = list_of_coords[node1]
    coords2 = list_of_coords[node2]
    x = coords1[0]-coords2[0]
    y = coords1[1]-coords2[1]
    return (x,y)

def find_nearest_node(coordinates, index):
    nearest = tuple(index.nearest(coordinates, 1))
    nearest_node = nearest[0]
    return nearest_node

def bigger_bbox(bb):
    diff_v = (bb[2]- bb[0])*0.5

    diff = (bb[3]- bb[1])
    diff_g =diff*0.1
    diff_g_kosyak =diff*0.4

    d = -1
    bb= list(bb)
    for x in range(len(bb)):
        if x%2==0:
            bb[x] = bb[x]+diff_v*d
        else:
            if x == 1:
                bb[x] = bb[x]+diff_g_kosyak*d
                d = 1
            else:
                bb[x] = bb[x]+diff_g*d
    return tuple(bb)

def get_path(list_of_edges, dataset, param):
    if param != 'weight':
        data = dataset[dataset.id.isin(list_of_edges)]
        data = data.rename(columns={'color_%s'%param:'color'})
        data = data[['geometry','color']]
        return data.to_json()
    else:
        data = dataset[dataset.id.isin(list_of_edges)]
        return data.to_json()
    
def get_response(list_of_edges, dataset, start, param):
    if param != 'weight':
        data = dataset[dataset['id'].isin(list_of_edges)]
        data = data.rename(columns={'color_%s'%param:'color'})
        length = round(data['len'].values.sum()/1000,2)
        time = int(data['time'].values.sum())
        data = data[['geometry','color']]
        bbox = data.total_bounds
        bbox = bigger_bbox(bbox)
        data = data.to_json()
        json_completer = start+(length,time,param)+bbox+(data,)
        answer = """{"start":[%f,%f],"length":%f,"time":%i,"type":"%s","zoom":{"sw":[%f,%f],"ne":[%f,%f]},"geom":%s}"""%json_completer
        return answer
    else:
        data = dataset[dataset['id'].isin(list_of_edges)]
        length = round(data['len'].values.sum()/1000,2)
        time = int(data['time'].values.sum())
        bbox = data.total_bounds
        bbox = bigger_bbox(bbox)
        data = data.to_json()
        json_completer = start+(length,time,param)+bbox+(data,)
        answer = """{"start":[%f,%f],"length":%f,"time":%i,"type":"%s","zoom":{"sw":[%f,%f],"ne":[%f,%f]},"geom":%s}"""%json_completer
        return answer

def distance(long p1, long p2, dict coords):
    cdef float x1,x2,y1,y2
    x1,y1 = coords[p1]
    x2,y2 = coords[p2]
    return (((x2-x1)**2+(y2-y1)**2)**0.5)*10

def neighs_iter(key, g):
    for x in g[key].items():
        yield x

def bidirectional_astar(G, source_coords, 
                        target_coords, heuristic,
                        spatial_index, dataset, 
                        coords, additional_param='weight'):
    
    nod = tuple([find_nearest_node(x, spatial_index) for x in [source_coords, target_coords]])
    source,target = nod
    start = coords[source]
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
            if v is not None and explored[1-d][v] is not None:
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
            return get_response(finalpath, dataset, start, additional_param)
        
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
    raise Exception('Path between given nodes does not exist.')

            
def composite_request(G, source_coords, target_coords, heuristic, spatial_index, dataset, coords):
    green_route =  bidirectional_astar(G, source_coords, target_coords, 
                                       heuristic, spatial_index, dataset, coords, additional_param = 'green')
    noisy_route = bidirectional_astar(G, source_coords, target_coords, 
                                       heuristic, spatial_index, dataset, coords, additional_param = 'noise')
    air_route = bidirectional_astar(G, source_coords, target_coords, 
                                       heuristic, spatial_index, dataset, coords, additional_param = 'air')
    answer = """[%s, %s, %s]"""%(green_route,noisy_route,air_route)
    return answer

def _connect_paths(G, source_coords, 
                        target_coords, heuristic,
                        spatial_index, dataset, 
                        coords, avoid, additional_param='weight'):
    
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
            if v is not None and explored[1-d][v] is not None:
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
            
            return finalpath
        
        if v in explored[d]:
            continue
        
        explored[d][v] = parent
        edge_parent[d][v] = edge
        
        for neighbor, w in neighs_iter(v, G):
            if len(G[neighbor])==1:
                continue
            if neighbor in explored[d]:
                continue
            if neighbor in avoid:
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
    raise Exception('Path between given nodes does not exist.')

def beautiful_path(G, source_coords, heuristic, spatial_index, dataset, coords, 
                   cutoff, additional_param = 'weight', avoid = None, first_step = None): 
    
    source = find_nearest_node(source_coords, spatial_index)
    start = coords[source]
    dist =  {}
    paths =  {source:[]}
    node_paths =  {source:[source]}
    fringe = [] 
    seen =   {source:0}
    heappush(fringe, (0, 0, 0, source))
    finalpath = []
    weights = {}
    params = {}
    
    while fringe:
        (d, k, p, v) = heappop(fringe)
        if v in dist:
            continue
        dist[v] = d
        weights[v] = k
        params[v] = p
        
        for neighbor, w in neighs_iter(v, G):
            if avoid is not None:
                if neighbor in avoid:
                    continue
            cost = w.get('time',None)
            additional = w.get(additional_param,1)
            if cost is None:
                continue
            vu_dist = dist[v] + additional
            real_weight = weights[v] + cost
            param = params[v] + additional
            
            if real_weight > cutoff:
                continue
                
            if neighbor in dist:
                if vu_dist < dist[neighbor]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
                    
            elif neighbor not in seen or vu_dist < seen[neighbor]:
                seen[neighbor] = vu_dist
                heappush(fringe, (vu_dist, real_weight, param, neighbor))
                node_paths[neighbor] = node_paths[v] + [neighbor]
                paths[neighbor] = paths[v] + [w.get('id',1)]
                params[neighbor] = params[v] + additional
                
    er =  0.8*cutoff
    par = {}
    for x in paths.keys():
        if weights[x] > er:
            #del paths[x]
            #del node_paths[x]
        #else:
            par[x] = params[x]

    par = sorted(par, key=par.get, reverse = False)
        
    if first_step == None:
        
        best = par.pop()
        path1 = paths[best]
        #av1 = int(len(node_paths[best])*0.02)
        first = get_vector(best, source, coords)
        
        second_step = beautiful_path(G, coords[best], heuristic, spatial_index, dataset, coords, 
                       cutoff, additional_param, avoid = node_paths[best][:-7], first_step = first)
        
        target_coords = coords[second_step[1]]
        path2 = second_step[0]
        second_step = second_step[2]
        #av2 = int(len(second_step)*0.02)
        to_avoid = node_paths[best][7:]+second_step[:-7]
        
        path3 = _connect_paths(G,  target_coords, source_coords, heuristic, spatial_index, dataset, 
                                                       coords, to_avoid, additional_param)
        
        return get_response(path1+
                            path2+
                            path3, dataset, start, additional_param)
    
    else:
        #while params:
        best = par.pop()
        if best == source:
            best = par.pop()
            #best_vect = get_vector(source, best, coords)
            #if pi*0.15 < abs(get_circ(best_vect, first_step)) < pi*0.85:
             #   break
        return paths[best], best, node_paths[best]
    
def beautiful_composite_request(G, source_coords, heuristic, spatial_index, dataset, coords, cutoff):
    green_route =  beautiful_path(G, source_coords, heuristic, spatial_index, dataset, 
                                       coords, cutoff, additional_param = 'green')
    noisy_route = beautiful_path(G, source_coords, heuristic, spatial_index, dataset, 
                                       coords, cutoff, additional_param = 'noise')
    air_route = beautiful_path(G, source_coords, heuristic, spatial_index, dataset, 
                                       coords, cutoff, additional_param = 'air')
    answer = """[%s, %s, %s]"""%(green_route,noisy_route,air_route)
    return answer