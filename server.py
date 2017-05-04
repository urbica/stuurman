import os
import urllib
import pickle

from flask import Flask, request
import networkx as nx
from support import colorize, set_spatial_index, transform
import pyximport
pyximport.install()

from functions import bidirectional_astar, distance, composite_request
from get_back_func import beautiful_path, beautiful_composite_request

path = os.path.dirname(os.path.realpath(__file__))

with open(path+'/graphs/edges_green_noise_air.pickle','r') as f:
    edges = pickle.load(f)
with open(path+'/graphs/nodes_osm.pickle','r') as f:
    nodes = pickle.load(f)

coords={}
nodes = nodes[(nodes.id.isin(edges.source.values))|(nodes.id.isin(edges.target.values))]
for x in range(nodes.shape[0]):
    coord = (nodes.geometry.values[x].x,nodes.geometry.values[x].y)
    id = nodes.id.values[x]
    coords[id] = coord
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
    G.add_node(a)
    G.add_node(b)
    G.add_edge(a,b, {'weight':w, 'green':w*g, 'noise':w*n, 'air':w*air, 'id':edge_id})

spatial = set_spatial_index(coords)
edges = edges[['id','color_green','color_noise','color_air','geometry', 'len', 'time']]
G = G.adj

app = Flask(__name__)

@app.route('/ping')
def ping():
    return 'Pong'

@app.route('/')
def index():
    page = urllib.urlopen(path+'/static/page.html').read()
    return page

@app.route('/ws', methods=['POST'])
def walking():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    return composite_request(G, coords1, coords2, distance, spatial, edges, coords)

@app.route('/shortest', methods=['POST'])
def shortest_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    return bidirectional_astar(G, coords1, coords2, distance, spatial, edges, coords)

@app.route('/green', methods=['POST'])
def green_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    return bidirectional_astar(G, coords1, coords2, distance, spatial, edges, coords, additional_param = 'green')

@app.route('/noise', methods=['POST'])
def noisy_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    return bidirectional_astar(G, coords1, coords2, distance, spatial, edges, coords, additional_param = 'noise')

@app.route('/air', methods=['POST'])
def air_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    return bidirectional_astar(G, coords1, coords2, distance, spatial, edges, coords, additional_param = 'air')

@app.route('/beautiful_path/green', methods=['POST'])
def beautiful_path_green_route():
    keys = request.get_json()
    coords = [keys['x'], keys['y']]
    time = transform(keys['time'])
    return beautiful_path(G, coords1, distance, spatial, edges, coords, time, additional_param = 'green')

@app.route('/beautiful_path/noise', methods=['POST'])
def beautiful_path_noise_route():
    keys = request.get_json()
    coords = [keys['x'], keys['y']]
    time = transform(keys['time'])
    return beautiful_path(G, coords1, distance, spatial, edges, coords, time, additional_param = 'noise')

@app.route('/beautiful_path/air', methods=['POST'])
def beautiful_path_noise_route():
    keys = request.get_json()
    coords = [keys['x'], keys['y']]
    time = transform(keys['time'])
    return beautiful_path(G, coords, distance, spatial, edges, coords, time, additional_param = 'air')

@app.route('/beautiful_path', methods=['POST'])
def beautiful():
    keys = request.get_json()
    coords = [keys['x'], keys['y']]
    time = transform(keys['time'])
    return beautiful_composite_request(G, coords, distance, spatial, edges, coords, time)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
    print 'Running'
