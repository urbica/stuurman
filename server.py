import os, datetime, urllib, csv, datetime, json
import pandas as pd
from flask import Flask, request

#import pyximport
#pyximport.install()

from functions import bidirectional_astar, distance, composite_request, beautiful_path, beautiful_composite_request, check_point, isochrone_from_point

path = os.path.dirname(os.path.realpath(__file__))

beatiful_path_logger= path+'/logs/logs_get_back.csv'
bidirectional_astar_logger= path+'/logs/logs_a_b.csv'
coordinates_logs = path+'/logs/logs_coords.csv'

steps = json.load(open(path+'/static/steps_in_graph.json'))

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
    if check_point(coords1) and check_point(coords2):
        writeLog(bidirectional_astar_logger, [coords1[0],coords1[1],coords2[0],coords2[1], datetime.datetime.now(),imei,1])
        return composite_request(coords1, coords2)
    else:
        writeLog(bidirectional_astar_logger, [coords1[0],coords1[1],coords2[0],coords2[1], datetime.datetime.now(),imei,0])
        return point_in_polygon_error


@app.route('/shortest', methods=['POST'])
def shortest_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    if check_point(coords1) and check_point(coords2):
        return bidirectional_astar(coords1, coords2)
    else:
        return point_in_polygon_error

@app.route('/bike_path', methods=['POST'])
def bike_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    if check_point(coords1) and check_point(coords2):
        return bidirectional_astar(coords1, coords2, avoid_edges=steps)
    else:
        return point_in_polygon_error

@app.route('/green', methods=['POST'])
def green_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    if check_point(coords1) and check_point(coords2):
        return bidirectional_astar(coords1, coords2, additional_param = 'green')
    else:
        return point_in_polygon_error

@app.route('/noise', methods=['POST'])
def noisy_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    if check_point(coords1) and check_point(coords2):
        return bidirectional_astar(coords1, coords2, additional_param = 'noise')
    else:
        return point_in_polygon_error

@app.route('/air', methods=['POST'])
def air_route():
    keys = request.get_json()
    coords1 = [keys['x1'], keys['y1']]
    coords2 = [keys['x2'], keys['y2']]
    if check_point(coords1) and check_point(coords2):
        return bidirectional_astar(coords1, coords2, additional_param = 'air')
    else:
        return point_in_polygon_error

@app.route('/beautiful_path/green', methods=['POST'])
def beautiful_path_green_route():
    keys = request.get_json()
    coordinates = [keys['x'], keys['y']]
    time = keys['time']/4.0
    if check_point(coordinates):
        return beautiful_path(coordinates, time, additional_param = 'green')
    else:
        return point_in_polygon_error

@app.route('/beautiful_path/noise', methods=['POST'])
def beautiful_path_noise_route():
    keys = request.get_json()
    coordinates = [keys['x'], keys['y']]
    time = keys['time']/4.0
    if check_point(coordinates):
        return beautiful_path(coordinates, time, additional_param = 'noise')
    else:
        return point_in_polygon_error

@app.route('/beautiful_path/air', methods=['POST'])
def beautiful_path_air_route():
    keys = request.get_json()
    coordinates = [keys['x'], keys['y']]
    time = keys['time']/4.0
    if check_point(coordinates):
        return beautiful_path(coordinates, time, additional_param = 'air')
    else:
        return point_in_polygon_error

@app.route('/beautiful_path', methods=['POST'])
def beautiful():
    keys = request.get_json()
    coordinates = [keys['x'], keys['y']]
    time = keys['time']/4.0
    imei = keys.get('imei',None)
    if check_point(coordinates):
        writeLog(beatiful_path_logger, [coordinates[0],coordinates[1],time*4, datetime.datetime.now(),imei,1])
        return beautiful_composite_request(coordinates, time)
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
    return isochrone_from_point(coordinates, start_time, time)

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
    print 'Running'
