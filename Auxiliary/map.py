"""
This file to generate html maps, one with markers indicating the situation of ambulance stations (map_stations.html),
and the other with markers pointing the locations of all the emergencies (map_emergencies.html).
"""

import folium
import folium.plugins as plugins

from incidents_generator import IncidentsGenerator


# Stations

LIST_OF_STATIONS = {
    'Lausanne': {'id': 0, 'lat': 46.523525, 'lng': 6.638332, 'canton': 'VD'},
    'Villars-Sainte-Croix': {'id': 1, 'lat': 46.567019, 'lng': 6.560686, 'canton': 'VD'},   #new
    'Nyon': {'id': 2, 'lat': 46.382640, 'lng':  6.226248, 'canton': 'VD'},    # new
    'Tour-de-Peiz': {'id': 3, 'lat': 46.457491, 'lng': 6.853541, 'canton': 'VD'},
    'Morges': {'id': 4, 'lat': 46.523000, 'lng': 6.501375, 'canton': 'VD'},
    'Yverdons-les-Bains': {'id': 5, 'lat': 46.772147, 'lng': 6.644941, 'canton': 'VD'},
    'Aigle': {'id': 6, 'lat': 46.312054, 'lng': 6.964270, 'canton': 'VD'},
    'Aubonne': {'id': 7, 'lat': 46.491782, 'lng': 6.388352, 'canton': 'VD'},    # new
    'Payerne': {'id': 8, 'lat': 46.819290, 'lng': 6.948423, 'canton': 'VD'},
    'Mézières': {'id': 9, 'lat': 46.593600, 'lng': 6.770645, 'canton': 'VD'},   # new
    'Pompales': {'id': 10, 'lat': 46.666568, 'lng': 6.503529, 'canton': 'VD'},  # new
    'L\'Abbaye': {'id': 11, 'lat': 46.649326, 'lng': 6.320007, 'canton': 'VD'},  # new
    'Sainte-Croix': {'id': 12, 'lat': 46.821755, 'lng': 6.502249, 'canton': 'VD'},  # new
    'Château d\'Oex': {'id': 13, 'lat': 46.478255, 'lng': 7.141267, 'canton': 'VD'},

    # NE
    'Neuchâtel': {'id': 14, 'lat': 46.996007, 'lng': 6.944550, 'canton': 'NE'},
    'La Chaux-de-Fonds': {'id': 15, 'lat': 47.087581, 'lng': 6.809231, 'canton': 'NE'},
    'Malviliers': {'id': 16, 'lat': 47.031679, 'lng': 6.868278, 'canton': 'NE'},
    'Val-de-Travers': {'id': 17, 'lat': 46.924731, 'lng': 6.632067, 'canton': 'NE'}

}

coords = [46.7785, 6.640833]

m = folium.Map(location=coords, zoom_start=9)

for station, information in LIST_OF_STATIONS.items():

    iframe = folium.IFrame(width=200, height=150)
    popup = folium.Popup(iframe, max_width=650)
    folium.Marker(
        location=[information['lat'], information['lng']], popup=popup,
        icon=plugins.BeautifyIcon(
            icon="arrow-down", icon_shape="marker",
            number=information['id']+1,
            border_color='black',
            background_color='white'
        )
    ).add_to(m)

    # print(information['id'], '&', station, '\\\\')

m.save('Auxiliary/map_stations.html')

# Emergencies

m = folium.Map(location=coords, zoom_start=9)

ig = IncidentsGenerator()
ig.idx = 0

for i in range(ig.num_incidents):
    folium.CircleMarker(location=ig.new_emergency()[2], radius=0.5).add_to(m)

m.save('Auxiliary/map_emergencies.html')
