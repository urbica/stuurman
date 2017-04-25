import os
import urllib
import pickle

from flask import Flask, request
import networkx as nx
from rtree import index
import pyximport
pyximport.install()

from functions import bidirectional_astar, distance

path = os.path.dirname(os.path.realpath(__file__))

with open(path+'/graphs/edges_osm.pickle','r') as f:
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

for x in range(edges.shape[0]):
    a = edges.source.values[x]
    b = edges.target.values[x]
    w = edges.cost.values[x]
    g = edges.green_ratio.values[x]
    if g>1:
        g = 1
    if g > 0.7:
        g = 0.7
    G.add_node(a)
    G.add_node(b)
    G.add_edge(a,b, {'weight':w, 'green':g})
def set_spatial_index(coordinates):
    p = index.Property()
    p.dimension = 2
    ind= index.Index(properties=p)
    for x,y in zip(coordinates.keys(),coordinates.values()):
        ind.add(x,y)
    return ind
def categorialize(x):
    if x > 0.8:
        return 3
    if x > 0.4:
        return 2
    else:
        return 1

spatial = set_spatial_index(coords)
G = G.adj

edges['color_type'] = edges.green_ratio.apply(categorialize)
edges = edges[['source', 'target', 'color_type','geometry']]

app = Flask(__name__)

@app.route('/ping')
def ping():
    return 'Pong'

@app.route('/')
def index():
    page = urllib.urlopen(path+'/static/page.html').read()
    return page

@app.route('/route', methods=['POST'])
def route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    return bidirectional_astar(G, coords1, coords2, distance, spatial, edges, coords)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    print 'Running'
