var mymap = L.map('mapid').setView([55.75, 37.65], 13);

L.tileLayer('https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token=pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw', {
  maxZoom: 18,
  id: 'mapbox.streets'
}).addTo(mymap);

function getColor(d) {
  switch (d) {
    case 3:
      return '#31a354';
      break;
    case 2:
      return '#addd8e';
      break;
    case 1:
      return '#f7fcb9';
      break;
  }
}

function getNoiseColor(d) {
  switch (d) {
    case 3:
      return '#EB944D';
      break;
    case 2:
      return '#F5C063';
      break;
    case 1:
      return '#FADA70';
      break;
  }
}

function style(feature) {
  return {
    color: getColor(feature.properties.color),
    weight: 6,
    opacity: 1
  };
}

function noiseStyle(feature) {
  return {
    color: getNoiseColor(feature.properties.color),
    weight: 6,
    opacity: 1
  };
}

function underStyle() {
  return {
    color: '#636363',
    weight: 8,
    opacity: 1
  };
}

const markerStart = L.marker([55.75, 37.67], { draggable: 'true' })
  .addTo(mymap);
const markerFinish = L.marker([55.77, 37.69], { draggable: 'true' })
  .addTo(mymap);

var url = '/green';
var url1 = '/noise';

var params = {
  x1: 37.69,
  y1: 55.75,
  x2: 37.69,
  y2: 55.77
};
var d = new FormData();

d.append("json", JSON.stringify(params));
var myLayer = L.geoJSON().addTo(mymap);
var underLayer = L.geoJSON().addTo(mymap);
var noiseLayer = L.geoJSON().addTo(mymap);
var underNoiseLayer = L.geoJSON().addTo(mymap);

fetch(url, {
  method: 'POST',
  mode: 'cors',
  headers: {
    'Access-Control-Allow-Origin': '*',
    'Content-Type': 'application/json',
    Accept: 'application/json'
  },
  body: JSON.stringify(params)
})
  .then(function (response) {
    return response.json();
  })
  .then(function (resp) {
    //return L.geoJSON(JSON.stringify(resp)).addTo(mymap);
    myLayer.addData(resp);
    underLayer.addData(resp);
    //myLayer.set_style(style);
  });
fetch(url1, {
  method: 'POST',
  mode: 'cors',
  headers: {
    'Access-Control-Allow-Origin': '*',
    'Content-Type': 'application/json',
    Accept: 'application/json'
  },
  body: JSON.stringify(params)
})
  .then(function (response) {
    return response.json();
  })
  .then(function (resp) {
    //return L.geoJSON(JSON.stringify(resp)).addTo(mymap);
    noiseLayer.addData(resp);
    underNoiseLayer.addData(resp);
    //myLayer.set_style(style);
  });
//var myLayer = L.geoJSON().addTo(mymap);
//myLayer.addData(geojsonFeature);

markerStart.on('dragend',
  function (event) {
    var marker = event.target;
    var startPosition = marker.getLatLng();
    var params = {
      x1: startPosition.lng,
      y1: startPosition.lat,
      x2: markerFinish.getLatLng().lng,
      y2: markerFinish.getLatLng().lat
    };

    var d = new FormData();
    d.append("json", JSON.stringify(params));
    fetch(url, {
      method: 'POST',
      mode: 'cors',
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json',
        Accept: 'application/json'
      },
      body: JSON.stringify(params)
    })
      .then(function (response) {
        return response.json();
      })
      .then(function (resp) {
        //return L.geoJSON(JSON.stringify(resp)).addTo(mymap);
        myLayer.clearLayers();
        underLayer.clearLayers();
        //myLayer.addData(resp);
        underLayer = L.geoJSON(resp, { style: underStyle }).addTo(mymap)
        myLayer = L.geoJSON(resp, { style: style }).addTo(mymap)
      });
    fetch(url1, {
      method: 'POST',
      mode: 'cors',
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json',
        Accept: 'application/json'
      },
      body: JSON.stringify(params)
    })
      .then(function (response) {
        return response.json();
      })
      .then(function (resp) {
        //return L.geoJSON(JSON.stringify(resp)).addTo(mymap);
        noiseLayer.clearLayers();
        underNoiseLayer.clearLayers();
        //myLayer.addData(resp);
        noiseLayer = L.geoJSON(resp, { style: underStyle }).addTo(mymap)
        underNoiseLayer = L.geoJSON(resp, { style: noiseStyle }).addTo(mymap)
      });
    console.log('startPosition', startPosition);
  }
);

markerFinish.on('dragend',
  function (event) {
    var marker = event.target;
    var finishPosition = marker.getLatLng();
    var params = {
      x1: markerStart.getLatLng().lng,
      y1: markerStart.getLatLng().lat,
      x2: finishPosition.lng,
      y2: finishPosition.lat
    };

    var d = new FormData();
    d.append("json", JSON.stringify(params));
    fetch(url, {
      method: 'POST',
      mode: 'cors',
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json',
        Accept: 'application/json'
      },
      body: JSON.stringify(params)
    })
      .then(function (response) {
        return response.json();
      })
      .then(function (resp) {
        //return L.geoJSON(JSON.stringify(resp)).addTo(mymap);
        myLayer.clearLayers();
        underLayer.clearLayers();
        //myLayer.addData(resp);
        underLayer = L.geoJSON(resp, { style: underStyle }).addTo(mymap)
        myLayer = L.geoJSON(resp, { style: style }).addTo(mymap)
      });
    fetch(url1, {
      method: 'POST',
      mode: 'cors',
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(params)
    })
      .then(function (response) {
        return response.json();
      })
      .then(function (resp) {
        //return L.geoJSON(JSON.stringify(resp)).addTo(mymap);
        noiseLayer.clearLayers();
        underNoiseLayer.clearLayers();
        //myLayer.addData(resp);
        noiseLayer = L.geoJSON(resp, { style: underStyle }).addTo(mymap)
        underNoiseLayer = L.geoJSON(resp, { style: noiseStyle }).addTo(mymap)
      });
    console.log('finishPosition', finishPosition);
  }
);
