import math
import networkx as nx
import dgl

EARTH_MEAN_RADIUS_METER = 6371008.7714

class SPoint:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng

    def __str__(self):
        return '({},{})'.format(self.lat, self.lng)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        # equal. Orginally is compared with reference. Here we change to value
        return self.lat == other.lat and self.lng == other.lng

    def __ne__(self, other):
        # not equal
        return not self == other

    def __hash__(self):
        return hash(str(self.lat) + " " + str(self.lng))


def same_coords(a, b):
    if a == b:
        return True
    else:
        return False


def distance(a,b):
    if same_coords(a, b):
        return 0.0
    delta_lat = math.radians(b.lat - a.lat)
    delta_lng = math.radians(b.lng - a.lng)
    h = math.sin(delta_lat / 2.0) * math.sin(delta_lat / 2.0) + math.cos(math.radians(a.lat)) * math.cos(
        math.radians(b.lat)) * math.sin(delta_lng / 2.0) * math.sin(delta_lng / 2.0)
    c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
    d = EARTH_MEAN_RADIUS_METER * c
    return d

class CandidatePoint(SPoint):
    def __init__(self, lat, lng, eid, error, offset, rate):
        super(CandidatePoint, self).__init__(lat, lng)
        self.eid = eid
        self.error = error
        self.offset = offset
        self.rate = rate

    def __str__(self):
        return '{},{},{},{},{},{}'.format(self.eid, self.lat, self.lng, self.error, self.offset, self.rate)

    def __repr__(self):
        return '{},{},{},{},{},{}'.format(self.eid, self.lat, self.lng, self.error, self.offset, self.rate)

    def __hash__(self):
        return hash(self.__str__())


# todo
def rate2gps(rn, rid, rate):
    # cords = np.array(rn.edgeCord[rid]).reshape(-1, 2).tolist()
    edges_dis = rn.get_edges_dis()
    (dis, cords) = edges_dis[rid]
    offset = dis * rate
    dist = 0  # temp distance for coords
    pre_dist = 0  # coords distance is smaller than offset

    if rate == 1.0:
        return SPoint(*cords[-1])
    if rate == 0.0:
        return SPoint(*cords[0])

    project_pt = SPoint(*cords[0])

    for i in range(len(cords) - 1):
        if i > 0:
            pre_dist += distance(SPoint(*cords[i - 1]), SPoint(*cords[i]))
        dist += distance(SPoint(*cords[i]), SPoint(*cords[i + 1]))
        if dist >= offset:
            if distance(SPoint(*cords[i]), SPoint(*cords[i + 1])) < 1e-6:  # zero segment length
                coor_rate = 0
            else:
                coor_rate = (offset - pre_dist) / distance(SPoint(*cords[i]), SPoint(*cords[i + 1]))
            project_pt = cal_loc_along_line(SPoint(*cords[i]), SPoint(*cords[i + 1]), coor_rate)
            break

    return project_pt


def gps2grid(pt, mbr, grid_size):
    """
    mbr:
    MBR class.
    grid size:
    int. in meter
    """
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    lat = pt.lat
    lng = pt.lng
    locgrid_x = int((lat - mbr.min_lat) / lat_unit) + 1
    locgrid_y = int((lng - mbr.min_lng) / lng_unit) + 1

    return locgrid_x, locgrid_y



def cal_loc_along_line(a, b, rate):
    """
    convert rate to gps location
    """
    lat = a.lat + rate * (b.lat - a.lat)
    lng = a.lng + rate * (b.lng - a.lng)
    return SPoint(lat, lng)


class RoadNetwork(nx.DiGraph):
    def __init__(self, g, edge_spatial_idx, edge_idx, edges_dis):
        super(RoadNetwork, self).__init__(g)
        # entry: eid
        self.edge_spatial_idx = edge_spatial_idx
        # eid -> edge key (start_coord, end_coord)
        self.edge_idx = edge_idx
        self.edges_dis = edges_dis
        

    def get_edge_idx(self):
        return self.edge_idx

    def get_edges_dis(self):
        return self.edges_dis
    def range_query(self, mbr):
        """
        spatial range query
        :param mbr: query mbr
        :return: qualified edge keys
        """
        eids = self.edge_spatial_idx.intersection((mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        return [self.edge_idx[eid] for eid in eids]


class MBR:
    def __init__(self, min_lat, min_lng, max_lat, max_lng):
        self.min_lat = min_lat
        self.min_lng = min_lng
        self.max_lat = max_lat
        self.max_lng = max_lng

    def contains(self, lat, lng):
        return self.min_lat <= lat < self.max_lat and self.min_lng <= lng < self.max_lng

def get_my_adj(file_path="data/road_adj.pkl"):
    import os
    print("extract_road_adj")
    G = nx.read_gml("data/road_graph.gml")
    g = dgl.from_networkx(G)
    return g
