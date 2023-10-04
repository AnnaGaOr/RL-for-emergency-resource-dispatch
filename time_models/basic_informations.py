""""
In this file, you can find all basic information we need.
"""

import os

# get the Root direction
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# MAX/MIN allowed LAT/LNG -> might be useful for data normalisation
LAT_MIN = 45.93
LAT_MAX = 47.35

LNG_MIN = 5.83
LNG_MAX = 8.32

# Some API keys...
# opencage_data_key
open_cage_data_key = 'bc0361a648464bacb84f7052348bc872'
# openroute_service_key
open_route_key = '5b3ce3597851110001cf62485a21e3b051144884b35cff86ef986219'

# pw and database name of the sia-remu database built by Simon
pw = 'siaremu2021HEIG!?'
db_name = 'history_data'


"""
The key of this dictionary is the location of an Emergency Medical Services
and the corresponding value are the id and the gps coordinate.
"""

NEIGHBORS_STATIONS = {
    'Lausanne': ['Villars-Sainte-Croix', 'Morges', 'Tour-de-Peiz', 'Mézières'],
    'Villars-Sainte-Croix': ['Pompales', 'Morges', 'Mézières', 'Lausanne'],
    'Nyon': ['Auboone', 'Morges'],
    'Tour-de-Peiz': ['Mézières', 'Lausanne', 'Aigle'],
    'Morges': [],
    'Yverdon-les-Bains': [],
    'Aigle': [],
    'Aubonne': [],
    'Payerne': [],
    'Mézières': [],
    'Pompales': [],
    'L\'Abbaye': [],
    'Sainte-Croix': [],
    'Château d\'Oex': [],

    'Neuchâtel': [],
    'La Chauds-de-Fonds': [],
    'Malviliers': [],
    'Val-de-Travers': []

}

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

"""
This list contains informations regarding the idaweb measures.

          Unité                                Description
fkl010h1  m/s                                  rafale (intégration 1 s); maximum horaire
brefarh0  No                                   éclairs lointains (éloignement 3 - 30 km); somme horaire
gre000h0  W/m²                                 rayonnement global; moyenne horaire
tre200hx  °C                                   température de l'air à 2 m du sol; maximum horaire
tre200hn  °C                                   température de l'air à 2 m du sol; minimum horaire
tre200h0  °C                                   température de l'air à 2 m du sol; moyenne horaire
brecloh0  No                                   éclairs proches (éloignement < 3 km); somme horaire
rre150hx  mm                                   précipitations; sommation sur 10 minutes, maximum horaire
rre150h0  mm                                   précipitations, somme horaire
ure200h0  %                                    humidité de l'air relative à 2 m du sol; moyenne horaire
hto000hs  cm                                   Hauteur de neige; valeur instantanée horaire
fkl010h0  m/s                                  vitesse du vent scalaire; moyenne horaire
su2000d0  h                                    durée d'ensoleillement; somme journalière
htoauths  cm                                   Hauteur de neige (mesurée automatiquement); valeur instantanée horaire
fkl002ix  m/s                                  vitesse du vent; moyenne sur 10 minutes maximale des dernières 2 heures
sre000d0  min                                  durée d'ensoleillement; somme journalière en minutes


"""
IDAWEB_MEASURES = ['wind gust; max over hour',     # 'fkl010h1'
                   'sun rays; mean over hour',     # 'gre000h0'
                   'temperature; max over hour',   # 'tre200hx'
                   'temperature; min over hour',     # 'tre200hn'
                   'temperature; mean over hour',    # 'tre200h0'
                   'precipitations; sum over 10 minutes, max in hour', # rre150hx
                   'num of thunders; at distance 3-30 km',   # brefarh0'
                   'air humidity; mean in hour',       # ure200h0'autom
                   'snow height',       # hto000hs'
                   'wind speed; mean over hour',        # fkl010h0'
                   'sum of sunlight over a day in hours',    # su2000d0'
                   'snow height; automatically measure', # htoauths'
                   'precipitations; sum over hour', # rre150h0'
                   'wind speed; mean over 10 minutes max in last 2 hours',  # fkl002ix'
                   'number of thunders; at distance < 3 km',  # brecloh0'
                   'sum of sunlight over a day in minutes',  # sre000d0'
                   'su200h0']   # 'su2000h0'

IDAWEB_MEASURES_2 = ['fkl010h1',
                   'gre000h0',
                   'tre200hx',
                   'tre200hn',
                   'tre200h0',
                   'rre150hx',
                   'brefarh0',
                   'ure200h0',
                   'hto000hs',
                   'fkl010h0',
                   'su2000d0',
                   'htoauths',
                   'rre150h0',
                   'fkl002ix',
                   'brecloh0',
                   'sre000d0',
                   'su2000h0']


"""
To build our predictive model, we tested a few different things. In particular,
we also tested a graph neural network. Below, you cna 


This dictionary represents the edges of our graph.
Each is the edge id, and the value are the name of the start and end node.

We put a graph into two locations if according to our knowledge of this region
it would make sense to drive from station the start station to the end station directly with a car.
Obviously, this construction is subjective. Maybe we should try another approach?

As always, we still do not take the NE stations into account...
"""
EDGES_STATIONS = {
    0: ['Yverdons-les-Bains', 'Payerne'],
    1: ['Payerne', 'Moudon'],
    2: ['Moudon', 'Lausanne'],
    3: ['Moudon', 'Etagnières'],
    4: ['Etagnières', 'Lausanne'],
    5: ['Yverdons-les-Bains', 'Etagnières'],
    6: ['Etagnières', 'Crissier'],
    7: ['Crissier', 'Lausanne'],
    8: ['Lausanne', 'Tour-de-Peiz'],
    9: ['Tour-de-Peiz', 'Aigle'],
    10: ['Aigle', 'Château d\'Oex'],
    11: ['Lausanne', 'Morges'],
    12: ['Morges', 'Allamans'],
    13: ['Allamans', 'Vich'],
    14: ['L\'Orient', 'Vich'],
    15: ['L\'Orient', 'Allamans'],
    16: ['L\'Orient', 'Morges'],
    17: ['Morges', 'Crissier'],
    18: ['Crissier', 'Chavornay'],
    19: ['L\'Orient', 'Chavornay'],
    20: ['Chavornay', 'Yverdons-les-Bains']
}


"""
The key of this dictionary is the name of the ofrou station and the key are
its id and gps coordinate.

!! we need to convert the gps coordinate using the function gps_coordinate_into_decimal (see below)
before to use it.
"""
idaweb_station_dict = {

    # VD
    "AIG": [28, '6.55', '46.20'],
    "VDARP": [29, '7.01', '46.53'],
    "VDBAU": [30, '6.33', '46.48'],
    "BAU": [31, '6.31', '46.47'],
    "VDBER": [32, '6.42', '46.42'],
    "BEX": [33, '6.60', '46.15'],
    "VDAVS": [34, '7.01', '46.15'],
    "BIE": [35, '6.21', '46.31'],
    "FRE": [36, '6.35', '46.50'],
    "CNZ": [37, '6.45', '46.44'],
    "SLFCHA": [38, '7.09', '46.23'],
    "SLFCH2": [39, '7.10', '46.23'],
    "CHY": [40, '6.33', '46.42'],
    "VDCHX": [41, '7.08', '46.28'],
    "CHD": [42, '7.08', '46.29'],
    "CDM": [43, '7.06', '46.23'],
    "VDCOR": [44, '6.36', '46.42'],
    "COS": [45, '6.30', '46.37'],
    "VDCRA": [46, '6.12', '46.22'],
    "VDCRI": [47, '6.35', '46.33'],
    "VDCUL": [48, '6.44', '46.29'],
    "ECH": [49, '6.37', '46.39'],
    "VDFOL": [50, '6.47', '46.32'],
    "VDFEC": [51, '6.22', '46.29'],
    "VDGEN": [52, '6.13', '46.26'],
    "VDGOU": [53, '6.36', '46.40'],
    "GRY": [54, '7.04', '46.16'],
    "ABG": [55, '6.29', '46.45'],
    "SLF1LC": [56, '7.05', '46.23'],
    "CUE": [57, '6.05', '46.28'],
    "DOL": [58, '6.06', '46.25'],
    "VDPEU": [59, '7.04', '46.16'],
    "NABLAU": [60, '6.38', '46.31'],
    "LSN": [61, '6.39', '46.32'],
    "WSLLAB": [62, '6.39', '46.35'],
    "WSLLAF": [63, '6.39', '46.35'],
    "CHE": [64, '6.13', '46.36'],
    "VDLMT": [65, '6.39', '46.33'],
    "VDLSP": [66, '6.14', '46.36'],
    "VDSEP": [67, '7.03', '46.22'],
    "AVA": [68, '6.57', '46.27'],
    "BIO": [69, '6.16', '46.37'],
    "CHB": [70, '6.19', '46.40'],
    "DIA": [71, '7.12', '46.20'],
    "DIB": [72, '7.10', '46.21'],
    "SLF1LS": [73, '7.02', '46.21'],
    "VDLUC": [74, '6.51', '46.42'],
    "MAH": [75, '6.34', '46.44'],
    "VDMOI": [76, '7.01', '46.57'],
    "VDMOL": [77, '6.22', '46.34'],
    "MAC": [78, '6.31', '46.31'],
    "MDO": [79, '6.48', '46.41'],
    "CGI": [80, '6.14', '46.24'],
    "VDORN": [81, '6.32', '46.43'],
    "BOZ": [82, '6.33', '46.44'],
    "VDNOC": [83, '6.32', '46.41'],
    "ORO": [84, '6.51', '46.34'],
    "ORZ": [85, '6.41', '46.43'],
    "PAY": [86, '6.57', '46.49'],
    "PAA": [87, '6.54', '46.50'],
    "PAV": [88, '6.56', '46.50'],
    "PUY": [89, '6.40', '46.31'],
    "VDREN": [90, '6.56', '46.23'],
    "VDREV": [91, '6.26', '46.33'],
    "VDROL": [92, '6.20', '46.27'],
    "VDROM": [93, '6.38', '46.51'],
    "GESAV": [94, '6.08', '46.19'],
    "PRE": [95, '6.27', '46.29'],
    "TDG": [96, '6.44', '46.31'],
    "VDVAL": [97, '6.37', '46.48'],
    "VVI": [98, '6.23', '46.43'],
    "VDVAU": [99, '6.25', '46.42'],
    "VIT": [100, '6.43', '46.37'],
    "VDPPI": [101, '6.57', '46.24'],
    "YVN": [102, '6.39', '46.47'],

    "AUB": [103, '6.26', '46.49'],
    "LON": [104, '6.15', '46.30'],
    "WNSTDP": [105, '6.49', '46.28'],
    "WNSDOR": [106, '6.35', '46.31'],
    "WNSMOR": [107, '6.30', '46.30'],
    "WNSNYO": [108, '6.13', '46.22'],
    "WNSLDJ": [109, '6.15', '46.37'],
    "WNSROL": [110, '6.20', '46.27'],
    "WNSYVN": [111, '6.39', '46.47'],
    "VEV": [112, '6.49', '46.28'],
    "BOVDBA": [113, '6.24', '46.44'],
    "BOVDCR": [114, '6.55', '46.23'],
    "BOVDCH": [115, '6.41', '46.32'],
    "BOVDGR": [116, '6.38', '46.48'],
    "BOVDMO": [117, '6.55', '46.26'],
    "BOVDOU": [118, '6.34', '46.39'],
    "BONEPC": [119, '6.42', '46.51'],
    "BONEPA": [120, '6.40', '46.49'],
    "BOVDPS": [121, '6.54', '46.30'],
    "BOVDRA": [122, '6.57', '46.18'],
    "VDGLA": [123, '6.60', '46.54'],
    "WNSLAU": [124, '6.36', '46.31'],

    # NE
    "CBS": [0, '6.49', '46.50'],
    "WNSEST": [1, '6.50', '46.51'],
    "MMROM": [2, '6.55', '46.42'],
    "ROM": [3, '6.56', '46.42'],
    "BOV": [4, '6.53', '47.01'],
    "MMBOY": [5, '6.49', '46.57'],
    "CHM": [6, '6.59', '47.03'],
    "COG": [7, '6.45', '46.57'],
    "COU": [8, '6.40', '46.56'],
    "CRM": [9, '7.04', '47.03'],
    "BRL": [10, '6.37', '46.59'],
    "CDF": [11, '6.48', '47.05'],
    "BNE": [12, '6.43', '47.04'],
    "PDM": [13, '6.44', '46.60'],
    "NEU": [14, '6.57', '47.00'],
    "WNSSAB": [15, '6.59', '47.01'],
    "NEBEV": [16, '6.49', '46.56'],
    "NECER": [17, '6.54', '47.03'],
    "BONECO": [18, '6.52', '47.05'],
    "NECOU": [19, '6.40', '46.56'],
    "BONECR": [20, '7.02', '47.02'],
    "NELSE": [21, '6.48', '47.02'],
    "NELOC": [22, '6.46', '47.04'],
    "BONEPS": [23, '6.54', '47.01'],
    "BONEPN": [24, '6.54', '47.01'],
    "BONETH": [25, '7.02', '47.01'],
    "BOBE30": [26, '6.52', '46.58'],
    "BONESE": [27, '6.54', '46.59'],
}

"""
BAFU STATIONS
"""
BAFU_STATIONS = {
    'chaumont': {'id': 0, 'lat': 47.031588, 'lng': 6.960933},
    'bern': {'id': 1, 'lat': 46.948271, 'lng': 7.445193},
    'lausanne': {'id': 2, 'lat': 46.518951, 'lng': 6.635149},
    'montandon': {'id': 3, 'lat': 47.302812, 'lng': 6.840208},
    'payerne': {'id': 4, 'lat': 46.821116, 'lng': 6.936603},
    'sion': {'id': 5, 'lat': 46.223157, 'lng': 7.338148}

}

BAFU_NEIGHBORS = {
    'chaumont': 'bern',
    'bern': 'chaumont',
    'lausanne': 'payerne',
    'sion': 'lausanne',
    'payerne': 'bern',
    'montandon': 'chaumont',
}

BAFU_MEASURES = [
    ' pm10',
    ' pm25',
    ' o3',
    ' no2',
]

"""
The first number is the id of the station. And the others are the coordinate the ofrou
station in the strange swiss system coordinate
"""

ofrou_counting_point = {
    # VD
    "002": [2545269, 1158463],
    '024': [2511980, 1144261],
    '043': [2531160, 1153570],
    '064': [2538074, 1155288],
    '078': [2560432, 1181055],
    '083': [2560490, 1137020],
    '116': [2544482, 1150272],
    '127': [2530076, 1177266],
    '149': [2532454, 1158072],
    '172': [2564347, 1129758],
    '188': [2495109, 1146860],
    '206': [2536717, 1177902],
    '226': [2532793, 1156737],
    '234': [2542580, 1179197],
    '247':	[2542650, 1188430],
    '249':	[2500440, 1130360],
    '250':	[2562090, 1129100],
    '291':	[2533308, 1155358],
    '300':	[2521103, 1147823],
    '331':	[2558405, 1150217],
    '333':	[2533658, 1168571],
    '370':	[2552008, 1147300],
    '391':	[2550235, 1166922],
    '398':	[2582090, 1148520],
    '504':	[2505029, 1137991],
    '505':	[2507348, 1141026],
    '509':	[2515915, 1146927],
    '511':	[2528114, 1152062],
    '534':	[2533888, 1157277],
    '535':	[2541234, 1154049],
    '536':	 [2543208, 1151819],
    '537':	[2555810, 1147195],
    '538':	[2557346, 1145275],
    '608':	[2572414, 1136237],
    '743':	[2532720, 1157030],
    '744':	[2532653, 1157011],
    '779':	[2519559, 1176075],

    # NE
    '059': [2562400, 1205480],
    '112':	[2569036, 1207857],
    '121': [2556534, 1201494],
    '173':	[2525793, 1195061],
    '174':	[2545384, 1211346],
    '189':	[2549025, 1213210],
    '190':	[2545750, 1201074],
    '212':	[2558000, 1212670],
    '218':	[2569000, 1210107],
    '671':	[2548250, 1192238],
    '672':	[2551245, 1195850],
    '673':	[2552840, 1197720],
    '674':	[2554187, 1199737],
    '675':	[2559685, 1203680],
    '676':	[2563876, 1206061],
    '677':	[2559096, 1206513],
    '799':	[2558325, 1203250],
    '801':	[2567508, 1207039],
    '802':	[2567479, 1207038],
    '814':	[2567945, 1207070],
    '815':	[2566600, 1206997]
}

"""
For each EMS, we need to assign the same number of ofrou and idaweb station
(the input in the models needs to always have the same dimension). 

For each station (the key), we need to assign 3 ofrou stations and 3 idaweb stations.
To make this choice we try to use the stations which are the closes to the EMS station
and with the less of missing data are possible.

Obviously all these choices are questionable...
"""
ASSIGNED_STATIONS = {
    'Lausanne': {
        'ofrou': [64, 2, 535],
        'idaweb': ['NABLAU', 'PUY', 'WSLLAF'],
        'bafu': ['lausanne']
    },
    'Villars-Sainte-Croix': {
        'ofrou': [291, 534, 149],
        'idaweb': ['NABLAU', 'WSLLAF',  'COS'],
        'bafu': ['lausanne']
    },
    'Tour-de-Peiz': {
        'ofrou': [537, 538, 331],
        'idaweb': ['VEV', 'ORO', 'AVA'],
        'bafu': ['lausanne']
    },
    'Morges': {
        'ofrou': [43, 511, 300],
        'idaweb': ['PRE', 'NABLAU', 'COS'],
        'bafu': ['lausanne']
    },
    'Nyon': {
        'ofrou': [505, 504, 509],
        'idaweb': ['VDCRA', 'GESAV','DOL'],
        'bafu': ['lausanne']
    },
    'Yverdons-les-Bains': {
        'ofrou': [206, 234, 247],
        'idaweb': ['FRE', 'MAH', 'VDVAL'],
        'bafu': ['payerne']
    },
    'Aigle': {
        'ofrou': [172, 608, 83],
        'idaweb': ['AIG', 'CDM', 'DIA'],
        'bafu': ['sion']

    },
    'Aubonne': {
        'ofrou': [300, 509, 24],
        'idaweb': ['VDFEC', 'BIE', 'PRE'],
        'bafu': ['lausanne'],
    },
    'Payerne': {
        'ofrou': [78, 234, 391],
        'idaweb': ['PAY', 'PAA', 'VDMOI'],
        'bafu': ['payerne']
    },
    'Mézières': {
        'ofrou': [391, 2, 535],
        'idaweb': ['ORO', 'VIT', 'VDLUC'],
        'bafu': ['payerne']
    },
    'Pompales': {
        'ofrou': [206, 333, 149],
        'idaweb': ['MAH', 'VIT', 'VDCOR'],
        'bafu': ['payerne']
    },
    'L\'Abbaye': {
        'ofrou': [188, 779, 127],
        'idaweb': ['LON', 'CHB', 'VDVAU'],
        'bafu': ['lausanne']
    },
    'Sainte-Croix': {
        'ofrou': [173, 247, 127],
        'idaweb': ['AUB', 'FRE', 'VDBAU'],
        'bafu': ['payerne']
    },
    'Château d\'Oex': {
        'ofrou': [398, 608, 172],
        'idaweb': ['CHD', 'CDM', 'SLFCH2'],
        'bafu': ['sion']
    },
    'La Chaux-de-Fonds': {
        'ofrou': [212, 174, 189],
        'idaweb': ['CDF', 'BRL', 'BNE'],
        'bafu': ['chaumont']
    },
    'Malviliers': {
        'ofrou': [212, 677, 799],
        'idaweb': ['CHM', 'NECER', 'CDF'],
        'bafu': ['chaumont']
    },
    'Val-de-Travers': {
        'ofrou': [190, 173, 674],
        'idaweb': ['BRL', 'VDROM', 'FRE'],
        'bafu': ['chaumont']
    },
    'Neuchâtel': {
        'ofrou': [676, 799, 121],
        'idaweb': ['NEU', 'CRM', 'CHM'],
        'bafu': ['chaumont']
    }
}


"""
For each idaweb and ofrou station present in ASSIGNED_STATIONS, we assign a 
list of neighbors station -> this might be useful to deal with missing data -> strategy 1
"""

NEIGHBORS = {
    676: [59, 815, 218],
    121: [674, 673, 799],
    64: [534, 535, 536, 116],
    2: [391, 78],
    535: [64, 536, 116, 534, 370, 538],   # !!
    504: [249, 505, 24],
    24: [505, 509, 300],
    212: [677, 675, 189],
    174: [189, 190, 212],
    189: [174, 190, 212],
    677: [212, 675, 59],
    799: [675, 59, 121],
    190: [173, 174],
    173: [190, 174],
    674: [189, 190],
    291: [226, 774, 149, 43],       # !!
    534: [64, 535, 536, 149],        # !!
    149: [333, 744, 226],
    249: [504, 505, 24],
    188: [504, 505, 24],
    505: [504, 24, 509],
    537: [370, 538, 331],
    538: [83, 537, 370],
    331: [537, 370, 538],
    43: [291, 226, 744],
    511: [300, 509, 24],
    300: [509, 511, 24],
    206: [247, 236, 333, 234],       # !!
    234: [206, 247, 333],
    247: [206, 234, 127],
    172: [608, 83, 250],
    608: [172, 250, 83],
    83: [250, 538, 537],
    333: [149, 206, 127, 234],       # !!!
    78: [391, 2, 234],
    391: [78, 2, 234],
    509: [300, 24, 505, 504, 249],        # !!
    779: [127, 188, 206, 391],       # !!
    127: [779, 188, 206],
    398: [608, 172],

    #
    'CDF': ['BONECO', 'NELOC', 'BNE', 'NELSE', 'PDM'],
    'BRL': ['PDM', 'NECOU', 'COG', 'NELSE', 'BNE'],
    'BNE': ['NELOC', 'CDF', 'BONECO', 'NECER', 'PDM'],
    'GESAV': ['VDCRA', 'WNSNYO', 'CGI', 'WNSROL', 'VDFEC'],
    'VDCRA': ['WNSNYO', 'CGI', 'WNSROL', 'VDFEC', 'VDGEN'],
    'AVA': ['BOVDMO', 'VDPPI', 'BOVDPS', 'VDREN', 'BOVDCR'],
    'VDVAL': ['BOVDGR', 'WNSYVN', 'BONEPA', 'CNZ', 'ORZ', 'VDORN'],
    'VDFEC': ['BIE', 'PRE', 'WNSMOR', 'CGI'],
    'VDVAU': ['CHB', 'VVI', 'BOVDBA', 'ABG', 'VDORN'],
    'VDBAU': ['BAU', 'VDBAU', 'FRE', 'VDROM', 'VDVAL'],
    'CHM': ['NECER', 'CRM', 'BONECR', 'BOV', 'BONEPN', 'BONETH'],
    'NECER': ['BONEPN', 'BOV', 'NELSE', 'PDM', 'CHM'],
    'VDROM': ['FRE', 'VDBAU', 'AUB', 'BAU', 'VDVAL'],
    'NEU': ['WNSSAB', 'BONETH', 'BONECR', 'BONESE', 'NEBEV'],
    'CRM': ['BONECR', 'BONETH', 'WNSSAB', 'NEU'],
    'NABLAU': ['PUY', 'LSN', 'VDLMT', 'WSLLAF', 'TDG', 'VIT'],
    'PUY': ['NABLAU', 'TDG', 'VDCUL', 'LSN', 'VDCRI', 'VIT', 'VDFOL'],
    'WSLLAF': ['VIT', 'ECH', 'BOVDOU', 'COS', 'ORO', 'NABLAU', 'TDG', 'VEV'],
    'COS': ['VDVAU', 'VDREV', 'WSLLAF', 'VIT', 'ECH', 'VDGOU', 'VDCRI'],
    'CGI': ['VDCRA', 'VDGEN', 'GESAV', 'VDFEC', 'WNSNYO', 'VDLSP', 'VDFEC', 'BIE'],
    'DOL': ['CUE', 'VDGEN', 'GESAV', 'LON', 'BIE', 'CGI', 'VDCRA'],
    'LON': ['BIE', 'VDGEN', 'DOL', 'CUE', 'VDMOL', 'VDREV', 'PRE'],
    'VEV': ['VDCUL', 'AVA', 'NABLAU', 'VDCUL', 'BOVDPS', 'BOVDMO', 'VDREN'],
    'ORO': ['VIT', 'VDFOL', 'TDG', 'VDLMT', 'VDLUC', 'TDG', 'WSLLAF'],
    'AIG': ['VDREN', 'BEX', 'VDAVS', 'BOVDRA', 'VDPEU', 'VDPPI', 'BOVDMO'],
    'PRE': ['VDFEC', 'CGI', 'BIE', 'LON', 'WNSMOR', 'MAC', 'VDCRI', 'COS'],
    'FRE': ['VDROM', 'AUB', 'VDBAU', 'VDVAL', 'MAH'],
    'AUB': ['ABG', 'BAU', 'VDBAU', 'FRE', 'VVI', 'CHB', 'MAH'],
    'PAA': ['PAV', 'PAY', 'VDARP', 'VDMOI', 'VDLUC', 'ORO'],
    'CDM': ['VDSEP', 'DIB', 'SLFCHA', 'CHD', 'VDCHX', 'SLFCH2', 'DIA'],
    'DIA': ['DIB', 'SLFCH2', 'SLFCHA', 'CDM', 'VDCHX', 'VDPEU'],
    'MAH': ['BOZ', 'VDORN', 'VDBER', 'COS', 'VDGOU', 'PAY'],
    'VIT': ['WSLLAF', 'ORO', 'ORZ', 'ECH'],
    'VDCOR': ['VDGOU', 'VDBER', 'VIT', 'MAH'],
    'PAY': ['PAV', 'PAA', 'VDMOI'],
    'VDMOI': ['VDGLA', 'VDARP', 'PAY', 'PAA'],
    'VDREV': ['COS', 'WSLLAF', 'VDCRI', 'BIE'],
    'BIE': ['LON', 'VDREV', 'VDMOL', 'COS'],
    'VDLUC': ['PAY', 'PAA', 'ORO'],
    'CHB': ['VVI', 'VDVAU', 'FRE'],
    'VDLSP': ['BIO', 'CHB', 'VDVAU', 'LON'],
    'CHD': ['VDCHX', 'CDM', 'DIA'],
    'SLFCH2': ['SLFCHA', 'DIA', 'CDM', 'DIB', 'VDSEP', 'CHD'],

}


def convert_ofrou_dict(ofrou_dict):
    """
    Convert the strange swiss coordinate of the ofrou dictionary
    into the more common gps coordinate
    """
    for key in ofrou_dict:
        value = ofrou_dict[key]
        est, west = swiss_coords_tolatlon(value[0], value[1])
        ofrou_dict[key] = [est, west]
    return ofrou_dict


def convert_idaweb_dict(ida_dict):
    """
    Convert the gps coordinate of the idaweb dict into gps in decimal format
    """
    for key in ida_dict:
        value = ida_dict[key]
        est, north = value[1], value[2]
        est, north = gps_coordinate_into_decimal(est, north)
        ida_dict[key] = [value[0], est, north]
    return ida_dict


def gps_coordinate_into_decimal(est, north):
    """
    convert gps coordinate in the form x.y which mean x°y' into gps coordinate of the form x.z°
    expected input is a string and expected output is a float
    """
    degree_est, second_est = est.split('.')
    degree_west, second_west = north.split('.')

    est_decimal = float(degree_est) + (float(second_est)/60)
    west_decimal = float(degree_west) + (float(second_west)/60)

    return est_decimal, west_decimal


def swiss_coords_tolatlon(E, N):
    """
    specific conversion from swiss system to GPS coordinates
    source : https://www.swisstopo.admin.ch/fr/cartes-donnees-en-ligne/
    calculation-services/navref.html
    :param E: (double) x in meters (0,0) in Bern
    :param N: (double) y in meters (0,0) in Bern
    :return: (double,double) latitude and longitude
    """
    if E == 0 or N == 0:
        return 0, 0
    # look if it is either MN95 or MN03 format
    # i.e. if it is MN03, E has 6 decimal numbers, otherwise 7
    r = E / 100000
    yp = (E - 2600000) / 1000000 if r > 10 else (E - 600000) / 1000000
    xp = (N - 1200000) / 1000000 if r > 10 else (N - 200000) / 1000000
    lambdap = 2.6779094 + 4.728982 * yp + 0.791484 * xp * yp + 0.1306 * yp * xp * xp - 0.0436 * yp * yp * yp
    phip = 16.9023892 + 3.238272 * xp - 0.270978 * yp * yp - 0.002528 * xp * xp - 0.0447 * yp * yp * xp - 0.0140 * xp * xp * xp
    return phip * 100./36, lambdap * 100./36