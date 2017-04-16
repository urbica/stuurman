from shapely import geometry
from scipy.spatial import ConvexHull
from math import pow
import networkx as nx
from heapq import heappop, heappush
from rtree import index
import geocoder
from shapely.ops import linemerge
import geopandas as gp
from itertools import count
from numpy import log, log10
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
def set_spatial_index(coordinates):
    p = index.Property()
    p.dimension = 2
    ind= index.Index(properties=p)
    for x,y in zip(coordinates.keys(),coordinates.values()):
        ind.add(x,y)
    return ind
def find_nearest_node(coordinates, index):
    if type(coordinates) == str:
        c = geocoder.yandex(coordinates).latlng
        coordinates = [float(c[x]) for x in [1,0]]
    nearest = list(index.nearest(coordinates, 1))
    nearest_node = nearest[0]
    return nearest_node
def get_path(list_of_nodes, dataset):
    #g = gp.GeoDataFrame()
    data = dataset[(dataset.source.isin(list_of_nodes))&(dataset.target.isin(list_of_nodes))]
    #print data.green_ratio.sum()
    #print data.cost.sum()
    #geo = linemerge(data.geometry.values)
    #g['geometry'] = [geo]
    return data.to_json()
def three_best(d, p, dataset):
    return [get_path(p[v], dataset) for v in d]
def bidirectional_dijkstra(G, source_coords, target_coords, spatial_index, dataset, weight = 'weight',  shortest=False, avoid = None):
    if type(source_coords) != int:
        nod = [find_nearest_node(x, spatial_index) for x in [source_coords, target_coords]]
        source,target = tuple(nod)
    else:
        source = source_coords
        target = target_coords
    if source == target: return (0, [source])
    pop = heappop
    push = heappush
    dists =  [{},                {}]
    paths =  [{source:[source]}, {target:[target]}] 
    fringe = [[],                []] 
    seen =   [{source:0},        {target:0} ]
    push(fringe[0], (0, source))
    push(fringe[1], (0, target))
    neighs = [G.neighbors_iter, G.neighbors_iter]
    G = G.adj
    finalpath = []
    dir = 1
    get_weight = lambda x: x.get('green',1) + x.get(weight,1)
    if shortest:
        get_weight = lambda x: x.get(weight,1)
    while fringe[0] and fringe[1]:
        dir = 1-dir
        (dist, v )= pop(fringe[dir])
        if v in dists[dir]:
            continue
        dists[dir][v] = dist 
        if v in dists[1-dir]:
            return get_path(finalpath, dataset)
        for w in neighs[dir](v):
            if len(G[w])==1:
                if w!=target:
                    continue
            if(dir==0):
                minweight= get_weight(G[v][w])
                vwLength = dists[dir][v] + minweight
            else:
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
                push(fringe[dir], (vwLength,w))
                paths[dir][w] = paths[dir][v]+[w]
                if w in seen[0] and w in seen[1]:
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))
    
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