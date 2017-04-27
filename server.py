import os
import urllib
import pickle

from flask import Flask, request
import networkx as nx
from support import transform_noise, transform_green, categorialize, categorialize_noise, set_spatial_index
import pyximport
pyximport.install()

from functions import bidirectional_astar, distance, composite_request

path = os.path.dirname(os.path.realpath(__file__))

with open(path+'/graphs/edges_green_noise.pickle','r') as f:
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

edges['color_green'] = edges.green_ratio.apply(categorialize)
edges.green_ratio= edges.green_ratio.apply(transform_green)
edges['noise_ratio'] = edges.noise_db.apply(lambda x: 45.0 if x < 45 else x)
edges.noise_ratio= edges.noise_ratio.apply(transform_noise)
edges['color_noise'] = edges.noise_ratio.apply(categorialize_noise)

for x in range(edges.shape[0]):
    a = edges.source.values[x]
    b = edges.target.values[x]
    w = edges.cost.values[x]
    g = edges.green_ratio.values[x]
    n = edges.noise_ratio.values[x]
    G.add_node(a)
    G.add_node(b)
    G.add_edge(a,b, {'weight':w, 'green':w*g, 'noise':w*n})

spatial = set_spatial_index(coords)
edges = edges[['source', 'target','color_green','color_noise','geometry']]
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

@app.route('/green', methods=['POST'])
def green_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    return bidirectional_astar(G, coords1, coords2, distance, spatial, edges, coords, 'green')

@app.route('/noise', methods=['POST'])
def noisy_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    return bidirectional_astar(G, coords1, coords2, distance, spatial, edges, coords, 'noise')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
    print 'Running'
