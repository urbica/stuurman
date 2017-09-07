import os
import urllib
import pickle
import csv
import json
import datetime
import pandas as pd

from flask import Flask, request
#from werkzeug.contrib.fixers import ProxyFix
import networkx as nx
from support import colorize, set_spatial_index
import pyximport

pyximport.install()
from functions import bidirectional_astar, distance, composite_request, beautiful_path, beautiful_composite_request, check_point, isochrone_from_point, find_nearest_node

path = os.path.dirname(os.path.realpath(__file__))

with open(path+'/graphs/edges_green_noise_air_restrictions.pickle','r') as f:
    edges = pickle.load(f)
with open(path+'/graphs/nodes_osm.pickle','r') as f:
    nodes = pickle.load(f)
with open(path+'/static/border.pickle','r') as f:
    border = pickle.load(f)

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
    if check_point(stop_coordinates, border):
        stop_node[find_nearest_node(stop_coordinates, spatial)] = stops['stop_id'].values[x]
        node_stop[stops['stop_id'].values[x]]=find_nearest_node(stop_coordinates, spatial)
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
    G.add_node(a)
    G.add_node(b)
    G.add_edge(a,b, {'weight':w, 'green':w*g, 'noise':w*n, 'air':w*air, 'id':edge_id, 'time':time})

edges = edges[['id','color_green','color_noise','color_air','geometry', 'len', 'time']]
G = G.adj

beatiful_path_logger= path+'/logs/logs_get_back.csv'
bidirectional_astar_logger= path+'/logs/logs_a_b.csv'
coordinates_logs = path+'/logs/logs_coords.csv'

def writeLog(file, data):
    with open(file,'a') as fi:
        logger = csv.writer(fi)
        logger.writerow(data)

point_in_polygon_error = '''{"error":1}'''

print 'now ready'

app = Flask(__name__)

@app.route('/ping')
def ping():
    return 'Pong'

@app.route('/')
def index():
    page = urllib.urlopen(path+'/static/page.html').read()
    return page

@app.route('/log_coords', methods=['POST'])
def log_coordinates():
    coordinates = request.get_json()
    lat = coordinates['lat']
    lon = coordinates['lon']
    imei = coordinates.get('imei', None)
    writeLog(coordinates_logs, [lat,lon,datetime.datetime.now(),imei])
    return 'ok'

@app.route('/ws', methods=['POST'])
def walking():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    imei = keys.get('imei',None)
    if check_point(coords1, border) and check_point(coords2, border):
        writeLog(bidirectional_astar_logger, [coords1[0],coords1[1],coords2[0],coords2[1], datetime.datetime.now(),imei,1])
        return composite_request(G, coords1, coords2, distance, spatial, edges, coords)
    else:
        writeLog(bidirectional_astar_logger, [coords1[0],coords1[1],coords2[0],coords2[1], datetime.datetime.now(),imei,0])
        return point_in_polygon_error


@app.route('/shortest', methods=['POST'])
def shortest_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    imei = keys.get('imei',None)
    if check_point(coords1, border) and check_point(coords2, border):
        writeLog(bidirectional_astar_logger, [coords1[0],coords1[1],coords2[0],coords2[1], datetime.datetime.now(),imei,1])
        return bidirectional_astar(G, coords1, coords2, distance, spatial, edges, coords)
    else:
        writeLog(bidirectional_astar_logger, [coords1[0],coords1[1],coords2[0],coords2[1], datetime.datetime.now(),imei,0])
        return point_in_polygon_error

@app.route('/green', methods=['POST'])
def green_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    imei = keys.get('imei',None)
    if check_point(coords1, border) and check_point(coords2, border):
        writeLog(bidirectional_astar_logger, [coords1[0],coords1[1],coords2[0],coords2[1], datetime.datetime.now(),imei,1])
        return bidirectional_astar(G, coords1, coords2, distance, spatial, edges, coords, additional_param = 'green')
    else:
        writeLog(bidirectional_astar_logger, [coords1[0],coords1[1],coords2[0],coords2[1], datetime.datetime.now(),imei,0])
        return point_in_polygon_error

@app.route('/noise', methods=['POST'])
def noisy_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    imei = keys.get('imei',None)
    if check_point(coords1, border) and check_point(coords2, border):
        writeLog(bidirectional_astar_logger, [coords1[0],coords1[1],coords2[0],coords2[1], datetime.datetime.now(),imei,1])
        return bidirectional_astar(G, coords1, coords2, distance, spatial, edges, coords, additional_param = 'noise')
    else:
        writeLog(bidirectional_astar_logger, [coords1[0],coords1[1],coords2[0],coords2[1], datetime.datetime.now(),imei,0])
        return point_in_polygon_error

@app.route('/air', methods=['POST'])
def air_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    imei = keys.get('imei',None)
    if check_point(coords1, border) and check_point(coords2, border):
        writeLog(bidirectional_astar_logger, [coords1[0],coords1[1],coords2[0],coords2[1], datetime.datetime.now(),imei,1])
        return bidirectional_astar(G, coords1, coords2, distance, spatial, edges, coords, additional_param = 'air')
    else:
        writeLog(bidirectional_astar_logger, [coords1[0],coords1[1],coords2[0],coords2[1], datetime.datetime.now(),imei,0])
        return point_in_polygon_error

@app.route('/beautiful_path/green', methods=['POST'])
def beautiful_path_green_route():
    keys = request.get_json()
    coordinates = [keys['x'], keys['y']]
    time = keys['time']/4.0
    imei = keys.get('imei',None)
    if check_point(coordinates, border):
        writeLog(beatiful_path_logger, [coordinates[0],coordinates[1],time*4, datetime.datetime.now(),imei,1])
        return beautiful_path(G, coordinates, distance, spatial, edges, coords, time, additional_param = 'green')
    else:
        writeLog(beatiful_path_logger, [coordinates[0],coordinates[1],time*4, datetime.datetime.now(),imei,0])
        return point_in_polygon_error

@app.route('/beautiful_path/noise', methods=['POST'])
def beautiful_path_noise_route():
    keys = request.get_json()
    coordinates = [keys['x'], keys['y']]
    time = keys['time']/4.0
    imei = keys.get('imei',None)
    if check_point(coordinates, border):
        writeLog(beatiful_path_logger, [coordinates[0],coordinates[1],time*4, datetime.datetime.now(),imei,1])
        return beautiful_path(G, coordinates, distance, spatial, edges, coords, time, additional_param = 'noise')
    else:
        writeLog(beatiful_path_logger, [coordinates[0],coordinates[1],time*4, datetime.datetime.now(),imei,0])
        return point_in_polygon_error

@app.route('/beautiful_path/air', methods=['POST'])
def beautiful_path_air_route():
    keys = request.get_json()
    coordinates = [keys['x'], keys['y']]
    time = keys['time']/4.0
    imei = keys.get('imei',None)
    if check_point(coordinates, border):
        writeLog(beatiful_path_logger, [coordinates[0],coordinates[1],time*4, datetime.datetime.now(),imei,1])
        return beautiful_path(G, coordinates, distance, spatial, edges, coords, time, additional_param = 'air')
    else:
        writeLog(beatiful_path_logger, [coordinates[0],coordinates[1],time*4, datetime.datetime.now(),imei,0])
        return point_in_polygon_error

@app.route('/beautiful_path', methods=['POST'])
def beautiful():
    keys = request.get_json()
    coordinates = [keys['x'], keys['y']]
    time = keys['time']/4.0
    imei = keys.get('imei',None)
    if check_point(coordinates, border):
        writeLog(beatiful_path_logger, [coordinates[0],coordinates[1],time*4, datetime.datetime.now(),imei,1])
        return beautiful_composite_request(G, coordinates, distance, spatial, edges, coords, time)
    else:
        writeLog(beatiful_path_logger, [coordinates[0],coordinates[1],time*4, datetime.datetime.now(),imei,0])
        return point_in_polygon_error

#isochrones
@app.route('/isochrones', methods=['POST'])
def mighty_isochrones():
    keys = request.get_json()
    coordinates = [keys['x'], keys['y']]
    start_time = keys['start_time']
    time = keys['time']/60.0
    return isochrone_from_point(G, coordinates, start_time, spatial, time, nodes, routes, routes_on_stops, node_stop, stop_node)

#app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
    print 'Running'
