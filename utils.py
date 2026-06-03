import os
import numpy as np
import geopy.distance
import time
import torch
from osgeo import ogr
from road import SPoint, distance, RoadNetwork, rate2gps, gps2grid, MBR, CandidatePoint
from spatial_func import project_pt_to_segment
import math
from datetime import date, timedelta
from datetime import datetime as dt
import scipy.sparse as sp
import torch
from rtree import Rtree
import networkx as nx
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05

# 道路类型权重映射
HIGHWAY_WEIGHTS = {
    'motorway': 1.2,
    'trunk': 1.1,
    'primary': 1.0,
    'secondary': 0.9,
    'tertiary': 0.8,
    'residential': 0.7,
    'unclassified': 0.7,
    'service': 0.6,
    'motorway_link': 1.1,
    'trunk_link': 1.0,
    'primary_link': 0.9
}

def get_constraint_mat(n_road, rn, search_dis=200, alpha=0.65,roadtype_weight=True):
    """创新约束矩阵：结合距离衰减和方向相似性"""
    constraint_mat = torch.zeros(n_road, n_road)
    
    # 预处理：计算所有道路的方向向量
    for u, v, data in rn.edges(data=True):
        coords = data['coords']
        if len(coords) < 2:
            data['direction'] = (0, 0)
            continue
            
        # 计算首尾点方向向量
        start = coords[0]
        end = coords[-1]
        dx = end.lng - start.lng
        dy = end.lat - start.lat
        norm = math.sqrt(dx**2 + dy**2)
        
        # 归一化并存储
        data['direction'] = (dx/norm, dy/norm) if norm > 1e-6 else (0, 0)
        data['highway_type'] = data.get('highway', 'secondary')  # 从GML获取道路类型
    
    # 为每条道路创建约束向量
    for u, v, data in rn.edges(data=True):
        coords = data['coords']
        start = coords[0]
        end = coords[-1]
        mid = ((start.lat + end.lat)/2, (start.lng + end.lng)/2)
        point = SPoint(mid[0], mid[1])
        
        # 获取当前道路属性
        current_type = data.get('highway_type', 'secondary')
        current_direction = data['direction']
#        w_type = HIGHWAY_WEIGHTS.get(current_type, 1.0)

        
#        cons_vec = _innovative_dis_prob_vec(
#            point, rn, 
#            {'search_dist': search_dis, 'id_size': n_road},
#            current_direction, w_type, alpha
#        )
#        constraint_mat[data['eid']] = cons_vec
        # 消融实验处理：无道路类型权重
        if not roadtype_weight:
            w_type = 1.0  # 中性权重
        else:
            w_type = HIGHWAY_WEIGHTS.get(current_type, 1.0)
        
        cons_vec = _optimized_dis_prob_vec(
            point, rn, 
            {'search_dist': search_dis, 'id_size': n_road},
            current_direction, w_type, alpha, roadtype_weight
        )
        constraint_mat[data['eid']] = cons_vec

    return constraint_mat

def _optimized_dis_prob_vec(gps, rn, params, current_dir, w_type, alpha, roadtype_weight):
    """创新概率向量计算：混合距离和方向特征"""
    cons_vec = torch.zeros(params['id_size']) + 1e-10
    candis = get_candidates(gps, rn, params['search_dist'])
    
    if candis is None:
        return cons_vec
    
    for candi_pt in candis:
        rid = candi_pt.eid
        
        # 在路网中查找候选道路数据
        edge_data = None
        for u, v, data in rn.edges(data=True):
            if data['eid'] == rid:
                edge_data = data
                break
        
        if edge_data is None:
            continue
        
        # 1. 距离衰减分量
        dist_component = exp_prob(10, candi_pt.error)
        
        # 2. 方向相似性分量
        candi_dir = edge_data.get('direction', (0, 0))
        dir_similarity = 0
        if current_dir != (0, 0) and candi_dir != (0, 0):
            cos_sim = current_dir[0]*candi_dir[0] + current_dir[1]*candi_dir[1]
            dir_similarity = (1 + cos_sim) / 2  # 映射到[0,1]
        
        # 3. 混合权重
        mixed_score = alpha * dist_component + (1-alpha) * dir_similarity
        
        # 4. 道路类型加权
        candi_type = edge_data.get('highway_type', 'secondary')
#        w_candi = HIGHWAY_WEIGHTS.get(candi_type, 1.0)
#        final_score = w_type * w_candi * mixed_score
        
#        cons_vec[rid] = final_score
        if not roadtype_weight:
            # 消融实验：无道路类型权重
            w_candi = 1.0
            final_score = mixed_score
        else:
            w_candi = HIGHWAY_WEIGHTS.get(candi_type, 1.0)
            # 增强主干道影响
            if candi_type in ['motorway', 'trunk', 'primary']:
                final_score = w_type * w_candi * mixed_score * 1.2
            else:
                final_score = w_type * w_candi * mixed_score
        
        cons_vec[rid] = final_score

    return cons_vec
#def get_constraint_mat(n_road, rn, search_dis=300):
#    constraint_mat = torch.zeros(n_road, n_road)
#    for u,v,data in rn.edges(data=True):
#        coords = data['coords']
        # print(data['eid'])
#        start = coords[0]
#        end = coords[-1]
#        mid = ((start.lat + end.lat) / 2, (start.lng + end.lng) / 2)
#        point = SPoint(mid[0], mid[1])
#        cons_vec = get_dis_prob_vec([point, None], rn, {'search_dist':search_dis, 'id_size': n_road} )
#        constraint_mat[data['eid']] = cons_vec
    # print(constraint_mat)
#    return constraint_mat

def get_dis_prob_vec(gps, rn, parameters):
    """
    Args:
    -----
    gps: [SPoint, tid]
    """
    cons_vec = torch.zeros(parameters['id_size']) + 1e-10
    candis = get_candidates(gps[0], rn, parameters['search_dist'])
    if candis is not None:
        for candi_pt in candis:
            # if candi_pt.eid in raw2new_rid_dict.keys():
            new_rid = candi_pt.eid
            prob = exp_prob(10, candi_pt.error)
            cons_vec[new_rid] = prob
    else:
        cons_vec = torch.ones(parameters['id_size'])
    return cons_vec

def exp_prob(beta, x):
    """
    error distance weight.
    """
    return  math.exp(-pow(x,2)/pow(beta,2))

def get_candidates(pt, rn, search_dist):
    """
    Args:
    -----
    pt: point STPoint()
    rn: road network
    search_dist: in meter. a parameter for HMM_mm. range of pt's potential road
    Returns:
    --------
    candidates: list of potential projected points.
    """
    candidates = None
    mbr = MBR(pt.lat - search_dist * LAT_PER_METER,
              pt.lng - search_dist * LNG_PER_METER,
              pt.lat + search_dist * LAT_PER_METER,
              pt.lng + search_dist * LNG_PER_METER)
    candidate_edges = rn.range_query(mbr)  # list of edges (two nodes/points)
    if len(candidate_edges) > 0:
        candi_pt_list = [cal_candidate_point(pt, rn, candidate_edge) for candidate_edge in candidate_edges]
        # refinement
        candi_pt_list = [candi_pt for candi_pt in candi_pt_list if candi_pt.error <= search_dist]
        if len(candi_pt_list) > 0:
            candidates = candi_pt_list
    return candidates

def cal_candidate_point(raw_pt, rn, edge):
    """
    Get attributes of candidate point
    """
    u, v = edge
    coords = rn[u][v]['coords']  # GPS points in road segment, may be larger than 2
    candidates = [project_pt_to_segment(coords[i], coords[i + 1], raw_pt) for i in range(len(coords) - 1)]
    idx, (projection, coor_rate, dist) = min(enumerate(candidates), key=lambda x: x[1][2])
    # enumerate return idx and (), x[1] --> () x[1][2] --> dist. get smallest error project edge
    offset = 0.0
    for i in range(idx):
        offset += distance(coords[i], coords[i + 1])  # make the road distance more accurately
    offset += distance(coords[idx], projection)  # distance of road start position and projected point
    if rn[u][v]['length'] == 0:
        rate = 0
        # print(u, v)
    else:
        rate = offset/rn[u][v]['length']  # rate of whole road, coor_rate is the rate of coords.
    return CandidatePoint(projection.lat, projection.lng, rn[u][v]['eid'], dist, offset, rate)

def load_rn_shp(path, is_directed=True):
    edge_spatial_idx = Rtree()
    edge_idx = {}
    g = nx.read_shp(path, simplify=True, strict=False)

    for n, data in g.nodes(data=True):
        data['pt'] = SPoint(n[1], n[0])
        if 'ShpName' in data:
            del data['ShpName']

    edges_dis = {}
    for u,v,data in g.edges(data=True):
        geom_line = ogr.CreateGeometryFromWkb(data['Wkb'])
        coords = []
        mycoords = []
        for i in range(geom_line.GetPointCount()):
            geom_pt = geom_line.GetPoint(i)
            mycoords.append((geom_pt[1],geom_pt[0]))
            coords.append(SPoint(geom_pt[1], geom_pt[0]))
        data['coords'] = coords
        data['length'] = sum([distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1)])
 # 新增：提取道路类型
        highway_type = 'secondary'  # 默认值
        if 'highway' in data:
            highway_type = data['highway']
        elif 'Json' in data:  # 尝试从JSON解析
            try:
                json_data = json.loads(data['Json'])
                highway_type = json_data.get('highway', 'secondary')
            except:
                pass
        data['highway_type'] = highway_type        
        env = geom_line.GetEnvelope()
        edge_spatial_idx.insert(data['eid'], (env[0], env[2], env[1], env[3]))
        edge_idx[data['eid']] = (u,v)
        edges_dis[data['eid']] = (data['length'], mycoords)
        del data['ShpName']
        del data['Json']
        del data['Wkt']
        del data['Wkb']
    return RoadNetwork(g, edge_spatial_idx, edge_idx, edges_dis)

def get_rn_grid(mbr, rn, grid_size):
    rn_grid = []
    edges = rn.get_edge_idx()

    max_lat = 0
    max_lng = 0
    for rid in edges.keys():
        cur_grid = []
        for rate in range(1000):
            r = rate / 1000
            gps = rate2gps(rn, rid, r)
            lat = gps.lat
            lng = gps.lng
            if lat > max_lat:
                max_lat = lat
            if lng > max_lng:
                max_lng = lng
            grid_x, grid_y = gps2grid(gps, mbr, grid_size)
            if len(cur_grid) == 0 or [grid_x, grid_y] != cur_grid[-1]:
                cur_grid.append([grid_x, grid_y])
        rn_grid.append(torch.tensor(cur_grid))
    print(f"max_lat lng are {max_lat} {max_lng}")
    return rn_grid


def to_sparse_tensor(dense_matrix):
    return torch.from_numpy(dense_matrix)
    coo = sp.coo_matrix(dense_matrix)

    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.FloatTensor(coo.data)
    shape = coo.shape

    sparse_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
    
    print(f"The shhhhhh is {type(sparse_tensor)}")
    return sparse_tensor

def date_range(date1, date2):
    # date1, date2 = '20160401', '20160428'
    datetime1 = dt.strptime(date1, '%Y%m%d')
    datetime2 = dt.strptime(date2, '%Y%m%d')
    days = (datetime2 - datetime1).days + 1
    date_list = [(datetime1 + timedelta(day)).strftime('%Y%m%d') for day in range(days)]
    return date_list


def time_difference(time1, time2):
    # format: '25/03/2016 00:00:04'
    # time_difference = time1 - time2
    return (dt.strptime(time1, '%d/%m/%Y %H:%M:%S') - dt.strptime(time2, '%d/%m/%Y %H:%M:%S')).total_seconds()


def df_to_csv(df, file_path, index=False):
    print('Saving to file at %s'%(file_path))
    if os.path.exists(file_path):
        temp_file_path = '%s_temp'%(file_path)
        df.to_csv(temp_file_path, index=index)
        os.system('rm %s'%(file_path))
        os.system('mv %s %s'%(temp_file_path, file_path))
    else:
        df.to_csv(file_path, index=index)
    print('Saved.')

    
def print_log(line, log_path):
    with open(log_path, 'a') as f:
        f.write(str(line)+'\n')
    
    
def round_time(t, interval=5):
    # t = '25/03/2016 12:26:45'
    # output: '25/03/2016 12:25:00'
    # interval: in minutes
    interval = interval * 60 # convert minutes to seconds
    datetime = dt.strptime(t, '%d/%m/%Y %H:%M:%S')
    new_datetime = dt.fromtimestamp(int(time.mktime(datetime.timetuple())) // interval * interval)
    return new_datetime.strftime('%d/%m/%Y %H:%M:%S')


def geodistance(coords_1, coords_2):
    return geopy.distance.great_circle(coords_1, coords_2).m


##### Visualization #####
##### The code below for displaying road segments, road network, and vehicle trajectories
#
# import folium
#
# class Point():
#     def __init__(self, latitude=None, longitude=None, time=None):
#         self.lat = latitude
#         self.lon = longitude
#         self.time = time
        
#     def __str__(self):
#         return '%s, %s'%(self.lat, self.lon, self.time)
    

# def get_bearing(p1, p2):    
#     # Returns compass bearing from p1 to p2
    
#     long_diff = np.radians(p2.lon - p1.lon)
    
#     lat1 = np.radians(p1.lat)
#     lat2 = np.radians(p2.lat)
    
#     x = np.sin(long_diff) * np.cos(lat2)
#     y = (np.cos(lat1) * np.sin(lat2) 
#         - (np.sin(lat1) * np.cos(lat2) 
#         * np.cos(long_diff)))
#     bearing = np.degrees(np.arctan2(x, y))
    
#     # adjusting for compass bearing
#     if bearing < 0:
#         return bearing + 360
#     return bearing


# def get_arrow(locations, color='#3388ff', size=6, n_arrows=3, road_id=''):
    
#     # get arrow for a road segment to indicate the direction
#     # locations e.g. [(1.3096, 103.9081), (1.3103, 103.9079)]
    
#     # creating point from our Point named tuple
#     p1 = Point(locations[0][0], locations[0][1])
#     p2 = Point(locations[1][0], locations[1][1])
    
#     # getting the rotation needed for our marker.  
#     # Subtracting 90 to account for the marker's orientation
#     # of due East(get_bearing returns North)
#     rotation = get_bearing(p1, p2) - 90
    
#     # get an evenly space list of lats and lons for our arrows
#     # note that I'm discarding the first and last for aesthetics
#     # as I'm using markers to denote the start and end
# #     arrow_lats = np.linspace(p1.lat, p2.lat, n_arrows + 2)[1:n_arrows+1]
# #     arrow_lons = np.linspace(p1.lon, p2.lon, n_arrows + 2)[1:n_arrows+1]
#     arrow_lat = p2.lat
#     arrow_lon = p2.lon
    
#     arrows = []
    
#     #creating each "arrow" and appending them to our arrows list
# #     for points in zip(arrow_lats, arrow_lons):
#     arrow = folium.RegularPolygonMarker(location=(arrow_lat, arrow_lon), 
#                   weight=1, color=color, fill_color=color, number_of_sides=3, 
#                   radius=size, rotation=rotation, popup='%s, %s, %s'%(arrow_lat, arrow_lon, road_id))
#     return arrow


# def display_road(start_lon, start_lat, end_lon, end_lat, color='#3388ff', weight=3, m=None, tiles='OpenStreetMap', road_id='', arrow=True):
#     center_lat = (start_lat + end_lat) / 2
#     center_lon = (start_lon + end_lon) / 2
    
#     # plot map
#     if m is None:
#         m = folium.Map(location=[center_lat, center_lon], zoom_start=20, tiles=tiles)
#     # add road line
#     folium.PolyLine([(start_lat, start_lon), (end_lat, end_lon)], color=color, weight=weight).add_to(m)
#     # add direction arrow to road
#     if arrow:
#         get_arrow([(start_lat, start_lon), (end_lat, end_lon)], color=color, road_id=road_id).add_to(m)
    
#     return m


# def display_roads(road_ids, road_df, color='#3388ff', m=None, tiles='OpenStreetMap', arrow=True):
#     # E.g. display_roads([103067603, 103106763], road_df)
    
#     # extract roads
#     df = road_df.set_index('road_id')
#     roads = df.loc[road_ids]
    
#     # locate the center of the map
#     min_lat = min(min(roads['start_lat']), min(roads['end_lat']))
#     max_lat = max(max(roads['start_lat']), max(roads['end_lat']))
#     min_lon = min(min(roads['start_lon']), min(roads['end_lon']))
#     max_lon = max(max(roads['start_lon']), max(roads['end_lon']))
#     center_lat = (min_lat + max_lat) / 2
#     center_lon = (min_lon + max_lon) / 2
    
#     # plot map
#     if m is None:
#         m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=tiles)
    
#     # add road
#     for _, road in roads.iterrows():
#         display_road(road['start_lon'], road['start_lat'], road['end_lon'], road['end_lat'], color=color, m=m, road_id=road.name, arrow=arrow)
    
#     return m


# def display_roads_heatmap(road_ids, road_df, colors=None, weights=None, m=None, tiles='OpenStreetMap', arrow=True):
#     # E.g. display_roads([103067603, 103106763], road_df, ['#ff0000', '#ffa500'], [3, 3])
    
#     # extract roads
#     df = road_df.set_index('road_id')
#     roads = df.loc[road_ids]
    
#     # locate the center of the map
#     min_lat = min(min(roads['start_lat']), min(roads['end_lat']))
#     max_lat = max(max(roads['start_lat']), max(roads['end_lat']))
#     min_lon = min(min(roads['start_lon']), min(roads['end_lon']))
#     max_lon = max(max(roads['start_lon']), max(roads['end_lon']))
#     center_lat = (min_lat + max_lat) / 2
#     center_lon = (min_lon + max_lon) / 2
    
#     # plot map
#     if m is None:
#         m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=tiles)
    
#     # add road
#     if colors is None: colors = ['#3388ff'] * len(road_ids)
#     if weights is None: weights = [3] * len(road_ids)
#     for (_, road), color, weight in zip(roads.iterrows(), colors, weights):
#         display_road(road['start_lon'], road['start_lat'], road['end_lon'], road['end_lat'], color=color, weight=weight, m=m, road_id=road.name, arrow=arrow)
    
#     return m


# def display_road_network(min_lat, max_lat, min_lon, max_lon, road_df, color='#3388ff', m=None, tiles='OpenStreetMap', arrow=True):
#     # E.g. display_road_network(1.310, 1.315, 103.90, 103.91, road_df, color='red', tiles='cartodbpositron')
    
#     # extract roads
#     df = road_df[(road_df['start_lat']>=min_lat) & (road_df['start_lat']<=max_lat) & 
#                  (road_df['end_lat']>=min_lat) & (road_df['end_lat']<=max_lat) & 
#                  (road_df['start_lon']>=min_lon) & (road_df['start_lon']<=max_lon) & 
#                  (road_df['end_lon']>=min_lon) & (road_df['end_lon']<=max_lon)]
#     road_ids = list(df['road_id'])
#     df = df.set_index('road_id')
#     roads = df.loc[road_ids]
    
#     # locate the center of the map
#     center_lat = (min_lat + max_lat) / 2
#     center_lon = (min_lon + max_lon) / 2
    
#     # plot map
#     if m is None:
#         m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=tiles)
    
#     # add road
#     for _, road in roads.iterrows():
#         display_road(road['start_lon'], road['start_lat'], road['end_lon'], road['end_lat'], color=color, m=m, arrow=arrow)
    
#     return m


# def display_trajectory(points, m=None, tiles='OpenStreetMap'):
    
#     min_lat, max_lat, min_lon, max_lon = (91, -90, 181, -181)
#     markers = []
#     for point in points:
#         min_lat = min(min_lat, point.lat)
#         max_lat = max(max_lat, point.lat)
#         min_lon = min(min_lon, point.lon)
#         max_lon = max(max_lon, point.lon)
#         markers.append(folium.Marker([point.lat, point.lon], popup='%s'%(point.time)))
    
#     # locate the center of the map
#     center_lat = (min_lat + max_lat) / 2
#     center_lon = (min_lon + max_lon) / 2
    
#     # plot map
#     if m is None:
#         m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=tiles)
    
#     # add point
#     for marker in markers:
#         marker.add_to(m)
    
#     return m


# def display_vehicle_raw_trajectory(vehicle_id, start_time, end_time, df, m=None, tiles='OpenStreetMap'):
#     # extract points of the vehicle
#     vehicle_df = df[(df['vehicle_id']==vehicle_id) &
#                     (df['time'].apply(lambda t: time_difference(t, start_time) >= 0)) &
#                     (df['time'].apply(lambda t: time_difference(end_time, t) >= 0))
#                    ].drop_duplicates()
#     points = []
#     for _, row in vehicle_df.iterrows():
#         points.append(Point(row['lat'], row['lon'], row['time']))
    
#     # display trajectory
#     m = display_trajectory(points, m=m, tiles=tiles)
    
#     return m


# def display_vehicle_matched_trajectory(vehicle_id, start_time, end_time, df, m=None, tiles='OpenStreetMap'):
#     # extract points of the vehicle
#     vehicle_df = df[(df['vehicle_id']==vehicle_id) &
#                     (df['time'].apply(lambda t: time_difference(t, start_time) >= 0)) &
#                     (df['time'].apply(lambda t: time_difference(end_time, t) >= 0))
#                    ].drop_duplicates()
#     points = []
#     for _, row in vehicle_df.iterrows():
#         points.append(Point(row['matched_lat'], row['matched_lon'], row['time']))
    
#     # display trajectory
#     m = display_trajectory(points, m=m, tiles=tiles)
    
#     return m
import matplotlib.pyplot as plt
import seaborn as sns
def plot_constraint_matrix(constraint_mat, rn, filename="constraint_matrix.png"):
    """可视化创新约束矩阵"""
    # 创建热力图
    plt.figure(figsize=(16, 14))

    # 计算矩阵值范围
    min_val = constraint_mat.min().item()
    max_val = constraint_mat.max().item()

    # 创建热力图
    ax = sns.heatmap(
        constraint_mat.cpu().numpy(),
        cmap="viridis",
        vmin=min_val,
        vmax=max_val,
        square=True,
        cbar_kws={"shrink": 0.8}
    )

    # 设置标题和标签
    plt.title("Innovative Road Constraint Matrix", fontsize=16)
    plt.xlabel("Road Segment ID", fontsize=12)
    plt.ylabel("Road Segment ID", fontsize=12)

    # 添加道路类型标记（对角线）
    road_types = []
    for _, _, data in rn.edges(data=True):
        road_type = data.get('highway', 'secondary')
        road_types.append(road_type[0].upper())  # 使用首字母

    # 每50条道路标注一次，避免过于密集
    for i in range(0, len(road_types), 50):
        road_type_char = road_types[i]
        ax.text(
            i + 0.5, i + 0.5, road_type_char,
            ha='center', va='center',
            color='white', fontsize=8, fontweight='bold'
        )

    # 添加图例说明
    legend_text = (
        "Road Type Legend:\n"
        "M: Motorway\nT: Trunk\nP: Primary\n"
        "S: Secondary\nR: Residential\nU: Unclassified"
    )
    plt.figtext(0.82, 0.15, legend_text,
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    # 保存高质量图片
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved constraint matrix visualization to {filename}")
