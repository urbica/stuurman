import os
import urllib
import pickle

from flask import Flask, request
import networkx as nx
from support import transform_poi, transform_green, categorialize, categorialize_poi, set_spatial_index
import pyximport
pyximport.install()

from functions import bidirectional_astar, distance, geocode

path = os.path.dirname(os.path.realpath(__file__))

with open(path+'/graphs/edges_plus.pickle','r') as f:
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

d = {}
indexes = []
for x in range(edges.shape[0]):
    s = edges['source'].values[x]
    t = edges['target'].values[x]
    weight = edges['cost'].values[x]
    if (s,t) not in d:
        d[(s,t)] = weight
    else:
        if d[(s,t)]>weight:
            d[(s,t)] = weight
        else:
            indexes.append(x)

edges = edges.drop(edges.index[indexes])

edges.poi = edges.poi.apply(transform_poi)
edges['color_green'] = edges.green_ratio.apply(categorialize)
edges['color_poi'] = edges.poi.apply(categorialize_poi)
edges.green_ratio= edges.green_ratio.apply(transform_green)
edges['mixed'] = edges.poi*edges.green_ratio
edges['color_mixed'] = edges.mixed.apply(categorialize)

for x in range(edges.shape[0]):
    a = edges.source.values[x]
    b = edges.target.values[x]
    w = edges.cost.values[x]
    g = edges.green_ratio.values[x]
    p = edges.poi.values[x]
    gP = g*p
    G.add_node(a)
    G.add_node(b)
    G.add_edge(a,b, {'weight':w, 'green':g, 'poi':p, 'mixed':gP})

spatial = set_spatial_index(coords)
edges = edges[['source', 'target', 'color_green','color_poi','color_mixed','geometry']]
G = G.adj

app = Flask(__name__)

@app.route('/ping')
def ping():
    return 'Pong'

@app.route('/geocode', methods=['POST'])
def translate():
    tx = request.get_json()
    tx = tx['address']
    print tx
    return geocode(tx)

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

@app.route('/route', methods=['POST'])
def route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    return bidirectional_astar(G, coords1, coords2, distance, spatial, edges, coords)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
    print 'Running'
