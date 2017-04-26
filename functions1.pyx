from shapely import geometry
from math import pow
import networkx as nx
from heapq import heappop, heappush
from rtree import index
import geocoder
from shapely.ops import linemerge
import geopandas as gp
from math import pow
from numpy import arccos, pi

def vector_dist(tuple vector):
    cdef float d
    d = vector[0]**2+vector[1]**2
    return d**0.5

def get_circ(tuple vector1, tuple vector2):
    cdef float scal, evklid1, evklid2, total_evklid, answer
    scal = vector1[0]*vector2[0]+vector1[1]*vector2[1]
    evklid1,evklid2  = [vector_dist(vect) for vect in (vector1, vector2)]
    total_evklid = evklid1*evklid2
    if all((evklid1, evklid2)):
        answer = scal/total_evklid
        if arccos(answer)>pi*0.4 and evklid1/evklid2>0.1:
            return 1
        else:
            return 0
    else:
        return 1

def get_vector(long node1,long node2, dict list_of_coords):
    cdef tuple coords1, coords2
    cdef float x, y
    coords1 = list_of_coords[node1]
    coords2 = list_of_coords[node2]
    x = coords1[0]-coords2[0]
    y = coords1[1]-coords2[1]
    return (x,y)

def find_nearest_node(coordinates, index):
    if type(coordinates) == str:
        c = geocoder.yandex(coordinates).latlng
        coordinates = [float(c[x]) for x in [1,0]]
    nearest = tuple(index.nearest(coordinates, 1))
    nearest_node = nearest[0]
    return nearest_node

def get_path(list_of_nodes,dataset):
    data = dataset[(dataset.source.isin(list_of_nodes))&(dataset.target.isin(list_of_nodes))]
    return data.to_json()

def get_w(float w, float g):
    if g>0.3:
        return w*0.5
    else:
        return w
    
def neighs_iterator(v, G):
    for x in G[v].keys():
        yield x

def bidirectional_dijkstra(G, source_coords, target_coords, 
                            spatial_index, dataset, tree, weight = 'weight',
                              shortest=0, avoid = None):
    if type(source_coords) != int:
        nod = [find_nearest_node(x, spatial_index) for x in [source_coords, target_coords]]
        source,target = tuple(nod)
    else:
        source = source_coords
        target = target_coords
    if source == target: return (0, [source])
    vector = get_vector(source, target, tree)
    back_vector = get_vector(target, source, tree)
    dists =  [{},                {}]
    paths =  [{source:[source]}, {target:[target]}]
    fringe = [[],                []]
    seen =   [{source:0},        {target:0}]
    heappush(fringe[0], (0, source))
    heappush(fringe[1], (0, target))
    finalpath = []
    dir = 1
    get_weight = lambda x: get_w(x.get(weight,1), x.get('green',1))
    if shortest:
        get_weight = lambda x: x.get(weight,1)
    while fringe[0] and fringe[1]:
        dir = 1-dir
        (dist, v)= heappop(fringe[dir])
        if v in dists[dir]:
            continue
        dists[dir][v] = dist 
        if v in dists[1-dir]:
            return get_path(finalpath, dataset)
        for w in neighs_iterator(v,G):
            if len(G[w])==1:
                if w!=target:
                    continue
            if(dir==0):
                w_vector = get_vector(source, w, tree)
                if get_circ(w_vector, vector):
                    continue
                minweight= get_weight(G[v][w])
                vwLength = dists[dir][v] + minweight
            else:
                w_vector = get_vector(target, w, tree)
                if get_circ(w_vector, back_vector):
                    continue
                minweight= get_weight(G[w][v])
                vwLength = dists[dir][v] + minweight 
            if avoid is not None:
                if w in avoid:
                    continue
            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                seen[dir][w] = vwLength
                heappush(fringe[dir], (vwLength,w))
                paths[dir][w] = paths[dir][v]+[w]
                if w in seen[0] and w in seen[1]:
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise Exception("No path between %s and %s." % (source, target))

def three_best(d, p, dataset):
    return [get_path(p[v], dataset) for v in d]

cdef check_similarity(list l,list l2, dict p):
    m = 0
    for l1 in l:
        i = 0.0
        for x in p[l1]:
            if x in l2:
                i+=1
        if i/len(l2)<=0.5:
            m+=1
    if m == len(l):
        return True
    
def beautiful_path(G, source_coords, spatial_index, dataset, weight = 'weight', cutoff = None, pred=None): 
    G_succ = G.succ if G.is_directed() else G.adj
    source = find_nearest_node(source_coords, spatial_index)
    dist =  {}
    paths =  {source:[source]}
    fringe = [] 
    seen =   {source:0}
    heappush(fringe, (0, 0, source))
    neighs = G.neighbors_iter
    finalpath = []
    weights = {}
    get_weight = lambda u, v, x: x.get(weight,1) + x.get('green',1)
    get_real_weight = lambda u, v, x: x.get(weight, 1)
    while fringe:
        (d, k, v) = heappop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        weights[v] = k
        for u, e in G_succ[v].items():
            cost = get_weight(v, u, e)
            if cost is None:
                continue
            vu_dist = dist[v] + get_weight(v, u, e)
            real_weight = weights[v] + get_real_weight(v, u, e)
            if cutoff is not None:
                if real_weight > cutoff:
                    continue
            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                heappush(fringe, (vu_dist, real_weight, u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)
    keys = sorted(seen, key=seen.get, reverse = True)
    approved_keys = [keys.pop(0)]
    i = 0
    while len(approved_keys)<=2:
        x = keys.pop(0)
        if check_similarity(approved_keys, paths[x], paths):
            approved_keys.append(x)
    return three_best(approved_keys, paths, dataset)
def walking_street(G, source_coords, spatial_index, dataset, weight = 'weight', cutoff = None, pred=None): 
    G_succ = G.succ if G.is_directed() else G.adj
    source = find_nearest_node(source_coords, spatial_index)
    dist =  {}
    paths =  {source:[source]}
    fringe = [] 
    seen =   {source:0}
    heappush(fringe, (0, 0, source))
    neighs = G.neighbors_iter
    finalpath = []
    weights = {}
    get_weight = lambda u, v, x: x.get(weight,1) + x.get('green',1)
    get_real_weight = lambda u, v, x: x.get(weight, 1)
    while fringe:
        (d, k, v) = heappop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        weights[v] = k
        for u, e in G_succ[v].items():
            cost = get_weight(v, u, e)
            if cost is None:
                continue
            vu_dist = dist[v] + get_weight(v, u, e)
            real_weight = weights[v] + get_real_weight(v, u, e)
            if cutoff is not None:
                if real_weight > cutoff/2.0:
                    continue
            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                heappush(fringe, (vu_dist, real_weight, u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)
    keys = sorted(dist, key=dist.get, reverse = True)
    approved_keys = [keys.pop(0)]
    i = 0
    while len(approved_keys)<2:
        x = keys.pop(0)
        if check_similarity(approved_keys, paths[x], paths):
            approved_keys.append(x)
    return three_best(approved_keys,
                     paths, dataset
                     )+[bidirectional_dijkstra(G, int(approved_keys[0]),int(approved_keys[1]),
                                                spatial_index, dataset)]