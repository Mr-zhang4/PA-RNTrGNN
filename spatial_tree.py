import numpy as np
from rtree import index
from road import SPoint, distance
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05


class RoadSpatialTree:
    def __init__(self, rn, dimensions=2):
        self.idx = index.Index(properties=index.Property(dimension=dimensions))
        self.road_data = {}
        self.topology = {}  # 存储拓扑关系
        
        # 构建空间索引和拓扑关系
        for eid, (u, v) in rn.edge_idx.items():
            coords = rn[u][v]['coords']
            start = coords[0]
            end = coords[-1]
            mid_lat = (start.lat + end.lat) / 2
            mid_lng = (start.lng + end.lng) / 2
            
            # 插入空间索引
            self.idx.insert(eid, (mid_lng, mid_lat, mid_lng, mid_lat))
            
            # 存储路段数据
            self.road_data[eid] = {
                'coords': coords,
                'center': SPoint(mid_lat, mid_lng),
                'endpoints': (u, v)
            }
        
        # 构建拓扑关系图
        G = rn.to_undirected()
        for eid in self.road_data:
            u, v = self.road_data[eid]['endpoints']
            neighbors = set()
            for node in [u, v]:
                neighbors |= set(G.neighbors(node))
            self.topology[eid] = neighbors
    
    def query_neighbors(self, eid, radius=300):
        """查询半径内的邻近路段"""
        center = self.road_data[eid]['center']
        # 计算半径对应的经纬度偏移
        lat_offset = radius * LAT_PER_METER
        lng_offset = radius * LNG_PER_METER
        
        # 查询矩形范围
        bbox = (center.lng - lng_offset, center.lat - lat_offset, 
                center.lng + lng_offset, center.lat + lat_offset)
        return list(self.idx.intersection(bbox))
    
    def get_road_center(self, eid):
        """返回路段中心点"""
        return self.road_data[eid]['center']
    
    def get_direction_vector(self, eid):
        """获取路段方向向量"""
        coords = self.road_data[eid]['coords']
        if len(coords) < 2:
            return np.zeros(2)
        start = np.array([coords[0].lng, coords[0].lat])
        end = np.array([coords[-1].lng, coords[-1].lat])
        vec = end - start
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-8 else np.zeros(2)
    
    def get_topology_neighbors(self, eid):
        """获取拓扑邻居"""
        return self.topology.get(eid, set())
