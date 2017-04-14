import cython
import networkx as nx
import heapq
from rtree import index
import geocoder
from shapely.ops import linemerge
import geopandas as gp
def find_nearest_node(coordinates, index):
    if type(coordinates) == str:
        c = geocoder.yandex(coordinates).latlng
        coordinates = [float(c[x]) for x in [1,0]]
    nearest = list(index.nearest(coordinates, 2))
    nearest_node = nearest[1]
    return nearest_node
def get_path(list_of_nodes, dataset):
    g = gp.GeoDataFrame()
    data = dataset[(dataset.FROMNODENO.isin(list_of_nodes))&(dataset.TONODENO.isin(list_of_nodes))]
    geo = linemerge(data.geometry.values)
    g['geometry'] = [geo]
    return g.to_json()
def bidirectional_dijkstra(G, source_coords,target_coords, spatial_index, dataset, weight = 'weight'):
    source = find_nearest_node(source_coords, spatial_index)
    target = find_nearest_node(target_coords, spatial_index)
    if source == target: return (0, [source])
    dists =  [{},                {}]
    paths =  [{source:[source]}, {target:[target]}] 
    fringe = [[],                []] 
    seen =   [{source:0},        {target:0} ]
    heapq.heappush(fringe[0], (0, source))
    heapq.heappush(fringe[1], (0, target))
    neighs = [G.neighbors_iter, G.neighbors_iter]
    finalpath = []
    dir = 1
    while fringe[0] and fringe[1]:
        dir = 1-dir
        (dist, v )= heapq.heappop(fringe[dir])
        if v in dists[dir]:
            continue
        dists[dir][v] = dist 
        if v in dists[1-dir]:
            return get_path(finalpath, dataset)
        for w in neighs[dir](v):
            #if len(G[w])==1:
                #continue
            if(dir==0):
                minweight=G[v][w].get(weight,1)
                vwLength = dists[dir][v] + minweight
            else:
                minweight=G[w][v].get(weight,1)
                vwLength = dists[dir][v] + minweight 
            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                seen[dir][w] = vwLength
                heapq.heappush(fringe[dir], (vwLength,w))
                paths[dir][w] = paths[dir][v]+[w]
                if w in seen[0] and w in seen[1]:
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise nx.NetworkXNoPath("No path between %s and %s." % (source, target))