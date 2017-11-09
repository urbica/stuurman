from heapq import heappop, heappush
from rtree import index
from itertools import count
from collections import deque
from numpy import arccos, pi, percentile
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon
import pickle, json, math, os
from rtree import index
import networkx as nx
import pandas as pd
import geopandas as gp

# Data Loading
def colorize(column):
    data = []
    p33 = percentile(column.values, 33)
    p66 = percentile(column.values, 66)
    for x in column.values:
        if x > p66:
            data.append(3)
            continue
        elif x > p33:
            data.append(2)
            continue
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

def check_point(coordinates):
    if border.contains(Point(coordinates)):
        return True
    else:
        return False

def find_nearest_node(coordinates):
    nearest = tuple(spatial.nearest(coordinates, 1))
    nearest_node = nearest[0]
    return nearest_node

path = os.path.dirname(os.path.realpath(__file__))

# with open(path+'/graphs/edges_green_noise_air_restrictions.pickle','r') as f:
#     edges = pickle.load(f)
# with open(path+'/graphs/nodes_osm.pickle','r') as f:
#     nodes = pickle.load(f)
edges = pd.read_pickle(path+'/graphs/edges_green_noise_air_restrictions.pickle')
edges = gp.GeoDataFrame(edges)
nodes = pd.read_pickle(path+'/graphs/nodes_osm.pickle')
nodes = gp.GeoDataFrame(nodes)
with open(path+'/static/border.pickle','r') as f:
    border = pickle.load(f)

steps = json.load(open(path+'/static/steps_in_graph.json'))
edges['steps'] = edges['id'].apply(lambda x: 1 if x in steps else 0)


with open(path+'/graphs/routes_data.json','r') as f:
    routes = json.load(f)
with open(path+'/graphs/routes_on_stop_data.json','r') as f:
    routes_on_stops = json.load(f)
stops = pd.read_csv(path+'/graphs/stops.txt')

for x,y in routes_on_stops.items():
    if type(x)==unicode:
        routes_on_stops[int(x)]=y
        del routes_on_stops[x]

print 'starting to collect graph'

nodes = nodes[(nodes['id'].isin(edges['source']))|(nodes['id'].isin(edges['target']))]

coords={}
for x in range(nodes.shape[0]):
    coord = (nodes['geometry'].values[x].x,nodes['geometry'].values[x].y)
    id = nodes['id'].values[x]
    coords[id] = coord
spatial = set_spatial_index(coords)

stop_node = {}
node_stop = {}
for x in range(len(stops)):
    stop_coordinates = stops['stop_lon'].values[x],stops['stop_lat'].values[x]
    if check_point(stop_coordinates):
        stop_node[find_nearest_node(stop_coordinates)] = stops['stop_id'].values[x]
        node_stop[stops['stop_id'].values[x]]=find_nearest_node(stop_coordinates)
print 'nodes-stops made'

G = nx.Graph()

edges.green_ratio= edges.green_ratio.apply(lambda x: 1 if x>1 else x)
edges['color_air'] = colorize(edges.air_ratio)
edges['color_green'] = colorize(edges.green_ratio)
edges['color_noise'] = colorize(edges.noise_ratio)

for x in range(edges.shape[0]):
    a = edges.source.values[x]
    b = edges.target.values[x]
    w = edges.cost.values[x]
    g = 1-edges.green_ratio.values[x]
    n = 1-edges.noise_ratio.values[x]
    air = 1-edges.air_ratio.values[x]
    edge_id = edges.id.values[x]
    time = edges.time.values[x]
    has_steps = edges['steps'].values[x]
    G.add_node(a)
    G.add_node(b)
    G.add_edge(a,b, {'weight':w, 'green':w*g, 'noise':w*n, 'air':w*air, 'id':edge_id, 'time':time, 'steps':has_steps})

edges = edges[['id','color_green','color_noise','color_air','geometry', 'len', 'time']]
G = G.adj

# Router
def check_similarity(l,l2):
    i =  len(set(l2)-set(l))
    i2 =  len(set(l)-set(l2))
    if i/len(l2) > 0.9 or i2/len(l2) > 0.9:
        return True
    return False

def vector_dist(vector):
    d = vector[0]**2+vector[1]**2
    return d**0.5

def get_circ(vector1, vector2):
    scal = vector1[0]*vector2[0]+vector1[1]*vector2[1]
    evklid1,evklid2  = [vector_dist(vect) for vect in (vector1, vector2)]
    total_evklid = evklid1*evklid2
    answer = scal/total_evklid
    return arccos(answer)

def get_vector(node1, node2):
    coords1 = coords[node1]
    coords2 = coords[node2]
    x = coords1[0]-coords2[0]
    y = coords1[1]-coords2[1]
    return (x,y)

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

def get_path(list_of_edges, param):
    if param != 'weight':
        data = edges[edges['id'].isin(list_of_edges)]
        data = data.rename(columns={'color_%s'%param:'color'})
        data = data[['geometry','color']]
        return data.to_json()
    else:
        data = edges[edges['id'].isin(list_of_edges)]
        return data.to_json()

def get_response(list_of_edges, start, param):
    data = edges[edges['id'].isin(list_of_edges)]
    if param != 'weight':
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
        length = round(data['len'].values.sum()/1000,2)
        time = int(data['time'].values.sum())
        bbox = data.total_bounds
        bbox = bigger_bbox(bbox)
        data = data.to_json()
        json_completer = start+(length,time,param)+bbox+(data,)
        answer = """{"start":[%f,%f],"length":%f,"time":%i,"type":"%s","zoom":{"sw":[%f,%f],"ne":[%f,%f]},"geom":%s}"""%json_completer
        return answer

def distance(p1, p2):
    x1,y1 = coords[p1]
    x2,y2 = coords[p2]
    return (((x2-x1)**2+(y2-y1)**2)**0.5)*10

def neighs_iter(key):
    for x in G[key].items():
        yield x

def bidirectional_astar(source_coords, target_coords, additional_param='weight', step_mode=False):

    nod = tuple([find_nearest_node(x) for x in [source_coords, target_coords]])
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
            return get_response(finalpath, start, additional_param)

        if v in explored[d]:
            continue

        explored[d][v] = parent
        edge_parent[d][v] = edge

        for neighbor, w in neighs_iter(v):
            if step_mode:
                if w.get('steps',None) == 1:
                    continue
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
                h = distance(neighbor, heu[d])

            enqueued[d][neighbor] = ncost, h
            e = w.get('id',1)
            heappush(queue[d], (ncost+h, neighbor, ncost, v, e))
    raise Exception('Path between given nodes does not exist.')


def composite_request(source_coords, target_coords):
    try:
        green_route =  bidirectional_astar(source_coords, target_coords, additional_param = 'green')
        noisy_route = bidirectional_astar(source_coords, target_coords, additional_param = 'noise')
        air_route = bidirectional_astar(source_coords, target_coords, additional_param = 'air')
        answer = """[%s, %s, %s]"""%(green_route, noisy_route, air_route)
        return answer
    except Exception as e:
        return '''{"error":0}'''

def _connect_paths(source_coords, target_coords, avoid, additional_param='weight'):

    nod = tuple([find_nearest_node(x) for x in [source_coords, target_coords]])
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

        for neighbor, w in neighs_iter(v):
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
                h = distance(neighbor, heu[d])

            enqueued[d][neighbor] = ncost, h
            e = w.get('id',1)
            heappush(queue[d], (ncost+h, neighbor, ncost, v, e))
    raise Exception('Path between given nodes does not exist.')

def beautiful_path(source_coords, cutoff, additional_param = 'weight', avoid = None, first_step = None):

    source = find_nearest_node(source_coords)
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

        for neighbor, w in neighs_iter(v):
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
        first = get_vector(source, best)

        second_step = beautiful_path(coords[best], cutoff, additional_param, avoid = node_paths[best][:-7], first_step = first)

        target_coords = coords[second_step[1]]
        path2 = second_step[0]
        second_step = second_step[2]
        #av2 = int(len(second_step)*0.02)
        to_avoid = node_paths[best][7:]+second_step[:-7]

        path3 = _connect_paths(target_coords, source_coords, to_avoid, additional_param)

        return get_response(path1+
                            path2+
                            path3, start, additional_param)

    else:
        del params[source]
        while params:
            best = par.pop()
            best_vect = get_vector(source, best)
            if pi*0.15 < get_circ(best_vect, first_step):
                break
        return paths[best], best, node_paths[best]

def beautiful_composite_request(source_coords, cutoff):
    try:
        green_route =  beautiful_path(source_coords, cutoff, additional_param = 'green')
        noisy_route = beautiful_path(source_coords, cutoff, additional_param = 'noise')
        air_route = beautiful_path(source_coords, cutoff, additional_param = 'air')
        answer = """[%s, %s, %s]"""%(green_route,noisy_route,air_route)
        return answer

    except Exception as e:
        return '''{"error":0}'''

# isochrones functions
def transform_time(x):
    hour, minute, _ = x.split(':')
    if hour[0]=='0':
        hour = int(hour[1])
    else:
        hour = int(hour)
    if minute[0]=='0':
        minute = int(minute[1])
    else:
        minute = int(minute)
    return hour+minute/60.0

def get_polygon(points):
    if len(points)<3:
        return None
    convex_hull = nodes[nodes['id'].isin(points)]['geometry'].values
    pp = [(x.x,x.y) for x in convex_hull]
    cent=(sum([p[0] for p in pp])/len(pp),sum([p[1] for p in pp])/len(pp))
    pp = sorted(pp, key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
    if len(pp)<3:
        return None
    poly = Polygon(pp)
    return poly

def find_next_stops(stop_id, start_time, current_time, cutoff):
    routes_to_observe = routes_on_stops[stop_id]
    response = {}
    end_time = start_time+cutoff
    for route_id, data in routes_to_observe.items():
        departure = data['time']
        if departure<current_time or departure>end_time:
            continue
        route_data = routes[route_id]
        stop_sequence = data['sequence']
        for sequence_id, stop_data in route_data.items():
            if sequence_id<=stop_sequence:
                continue
            stop_id = stop_data['stop_id']
            departure_time = stop_data['departure_time']
            if departure_time<departure:
                continue
            if departure_time>end_time:
                break
            weight = departure_time-start_time
            response[stop_id] = weight
    return response

def collect_polygons(list_of_polygons):
    geoms = []
    for poly_points in list_of_polygons:
        poly = get_polygon(poly_points)
        if poly is not None:
            geoms.append(poly)
    geoms = unary_union(geoms)
    g = gp.GeoDataFrame()
    g['geometry'] = [geoms]
    g = g.simplify(0.001)
    return g.to_json()

def isochrone_from_point(source_coords, start_time, cutoff):
    source = find_nearest_node(source_coords)
    start_time = transform_time(start_time)

    dist =  {}
    fringe = []
    seen =   {source:0}
    c = count()
    heappush(fringe, (0, next(c), source))

    polygons = []
    stops_to_observe=[]

    get_weight = lambda x: x.get('time', 1)/60.0
    get_stop = lambda x: stop_node.get(x, False)
    get_node = lambda x: node_stop.get(x, False)
    while True:
        polygon_points = []
        while fringe:
            d, _, v = heappop(fringe)
            if v in dist:
                continue # already searched this node.
            dist[v] = d
            for u, e in neighs_iter(v):
                cost = get_weight(e)
                vu_dist = dist[v] + get_weight(e)

                if vu_dist > cutoff:
                    polygon_points.append(u)
                    continue

                stop_id = get_stop(u)
                if stop_id:
                    current_time = start_time+vu_dist
                    next_stops= find_next_stops(stop_id, start_time, current_time, cutoff)
                    for stop_id, distance in next_stops.items():
                        w = get_node(stop_id)
                        if w:
                            stops_to_observe.append((w, distance))
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    heappush(fringe, (vu_dist, next(c), u))

        polygons.append(polygon_points)
        if stops_to_observe==[]:
            return collect_polygons(polygons)
        else:
            s,d = stops_to_observe.pop(0)
            heappush(fringe, (d, next(c), s))

