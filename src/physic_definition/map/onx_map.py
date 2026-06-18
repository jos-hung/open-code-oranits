import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import random
from .decas import *
from pyproj import Proj, Transformer
from geopy.distance import geodesic

import copy
from configs.systemcfg import map_cfg
import pickle
import os

plt.show = lambda: None #prevent show() from plt, if someone call it.


def node_to_point(node):
    wgs84 = 'epsg:4326'  # Hệ tọa độ địa lý WGS 84
    utm = 'epsg:32647'   # Hanoi zone
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)
    lat = node['y']
    lon = node['x']        
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        raise ValueError("invalid coordinate")        
    x, y = transformer.transform(lon, lat)  # Lưu ý thứ tự (lon, lat)
    return Point(x, y)

def calculate_edge_length(u, v, G):
    point_u = (G.nodes[u]['y'], G.nodes[u]['x'])
    point_v = (G.nodes[v]['y'], G.nodes[v]['x'])
    return geodesic(point_u, point_v).meters
   
def merge_nodes(G, sdistance):
    # this work remove the nearest nodes, improve the performance of other algorithm during searching in map
    nodes_to_remove = [] 

    distance_min = float('inf')
    change_dict = {}
    cnt = 0
    for u, v, data in G.edges(data=True):
        for u1, v1, data1 in G.edges(data=True):
            if u1 == u:
                continue
            # Kiểm tra khoảng cách giữa 2 node
            distance = calculate_edge_length(u, u1, G)
            if distance_min > distance:
                distance_min = distance
            if distance < sdistance:
                # Tính toán điểm chung mới giữa u và u1
                new_x = (G.nodes[u]['x'] + G.nodes[u1]['x']) / 2
                new_y = (G.nodes[u]['y'] + G.nodes[u1]['y']) / 2
                new_node = len(G.nodes) + 1  # Tạo node mới với ID mới
                edges_to_move = list(G.edges(u, data=True)) + list(G.edges(u1, data=True))
                nodes_to_remove.append(u)
                nodes_to_remove.append(u1)
                change_dict[cnt] = (new_node, new_x, new_y, copy.deepcopy(edges_to_move), distance)
                print("will delete nodes with distance {}, node 1 {}, node 2 {} because of too near".format(distance, node_to_point(G.nodes[u]), node_to_point(G.nodes[u1])) )
                cnt += 1
    G.remove_nodes_from(nodes_to_remove)
    for v in change_dict.keys():
        print("apply delete node: ", new_node, new_x, new_y)
        new_node, new_x, new_y, edges_to_move, distance = change_dict[v]        
        G.add_node(new_node, x=new_x, y=new_y)
        for u_old, v_old, edge_data in edges_to_move:
            if v_old != u1 and v_old != u:
                G.add_edge(new_node, v_old, **edge_data)
    return G

                
    
def get_segment(G):
    roads_info = {}  # Lưu trữ thông tin con đường
    cnt = 0
    for u, v, data in G.edges(data=True):
        start_point = node_to_point(G.nodes[u])
        end_point = node_to_point(G.nodes[v])
        
        # degree_u = G.degree(u)
        # print(degree_u)
        # edges_of_u = list(G.edges(u))
        # for edge in edges_of_u:
        #     if not edge in G.edges:
        #         print("loi")
        #         exit(0)
        road_name = data.get('name', 'unknown')
        if isinstance(road_name, list):
            road_name = road_name[0]  
        elif road_name == 'unknown':
            road_name = "unknown_" + str(cnt)  
        length = data.get('length', 0)
        segment_key = (start_point, end_point, length) 
        if road_name not in roads_info:
            roads_info[road_name] = []  
        roads_info[road_name].append(segment_key)  
        cnt += 1
    return roads_info


def create_segment(road_st, segs, generator):
    t_rs = []
    avg_speeds = []
    seg_objs=[]
    intersection_l = set()
    for seg in segs:
        if road_st == 0: 
            avg_speed = 20 #m/s
            t_r = 20 #tasks/s      
            status = 0 
            t_rs.append(t_r)     
            avg_speeds.append(avg_speed)    
        elif road_st ==1:
            #set a random select status for the segment with a higet priority for status 0:                
            p = [0.95, 0.05]
            status = generator.choice([0,1], p= p)
            avg_speed = 17 #m/s
            t_r = 30 #tasks/s
            t_rs.append(t_r)     
            avg_speeds.append(avg_speed)
        elif road_st ==2:
            avg_speed = 14 #m/h
            t_r = 40 #tasks/s
            p = [0.7, 0.25, 0.05]
            status = generator.choice([0,1,2], p= p)
            t_rs.append(t_r)     
            avg_speeds.append(avg_speed)  
        elif road_st ==3:
            avg_speed = 10 #m/s
            t_r = 50 #tasks/s
            p = [0.5, 0.25, 0.2, 0.05]
            status = generator.choice([0,1,2,3], p= p)
            t_rs.append(t_r)     
            avg_speeds.append(avg_speed)
        elif road_st==4:
            avg_speed = 5 #m/s
            t_r = 60 #tasks/s
            p = [0.2, 0.2, 0.2, 0.2, 0.2]
            status = generator.choice([0,1,2,3, 4], p= p)
            t_rs.append(t_r)     
            avg_speeds.append(avg_speed)
        segmen_obj = Segment(seg[0], seg[1], status, None, t_r, avg_speed, length=seg[2])
        intersection_l.add(seg[0])
        intersection_l.add(seg[1])
        avg_t_r = np.mean(t_rs)
        avg_speed_ = np.mean(avg_speeds)
        seg_objs.append(segmen_obj)
    return seg_objs, avg_t_r, avg_speed_, intersection_l

class HMap:
    def __init__(self, center_point, radius):
        map_folder = 'map'  
        file_path = os.path.join(map_folder, 'graph.pkl')
        
        if not map_cfg['fromfile']:
            custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
            # custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]'
            # custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential"]'
            # custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
            G = ox.graph_from_point(center_point, dist=radius, custom_filter=custom_filter, network_type='drive')
            # edges_to_remove = [(u, v, k) for u, v, k, data in G.edges(keys=True, data=True) if data.get('length', 0) < 10]
            # G.remove_edges_from(edges_to_remove)
            
            isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree < 2]
            G.remove_nodes_from(isolated_nodes)
            
            largest_component = max(nx.strongly_connected_components(G), key=len)
            self.G = G.subgraph(largest_component).copy()
            
            if not os.path.exists(map_folder):
                os.makedirs(map_folder)
            with open(file_path, 'wb') as f:
                pickle.dump(G, f)
        else:
            try:
                with open(file_path, 'rb') as f:
                    G = pickle.load(f)
            except:
                print("Error: If you don't have folder map with file graph.pkl. It leads you cannot run, please change setup in map_cfg to: 'fromfile:0'.")
                exit(0)
        
            self.G = G
        
        # fig, ax = ox.plot_graph(self.G, node_size=10, edge_linewidth=0.5, bgcolor='white', edge_color='blue')
        # node_pos = {node: (data['x'], data['y']) for node, data in self.G.nodes(data=True)}
        # nx.draw_networkx_nodes(self.G, node_pos, node_size=10, node_color='red', ax=ax)
        # plt.savefig("hanoi_map_.png", dpi=300)
        
        fig, ax = ox.plot_graph(
        self.G, 
        node_size=10, 
        edge_linewidth=0.5, 
        bgcolor='lightgray',  # Thay đổi màu nền thành xám nhạt
        edge_color='black'    # Thay đổi màu cạnh thành đen
        )
        node_pos = {node: (data['x'], data['y']) for node, data in self.G.nodes(data=True)}
        nx.draw_networkx_nodes(
            self.G, 
            node_pos, 
            node_size=10, 
            node_color='red', 
            ax=ax
        )
        plt.savefig("hanoi_map_.png", dpi=300)
        plt.close()
        
    def build_map(self, busy_status, busy_ps, busy, generator):
    
        self.road_infor = get_segment(self.G)
        segments, all_intersections, road_infor_change =self.update_map(generator, busy_status,busy_ps,busy)
        return  segments, road_infor_change, all_intersections

    def update_map(self, generator, busy_status,busy_ps,busy):
        print('map updating ...')
        segments = []
        all_intersections = set()
        road_infor_change = {}
        for road in self.road_infor.keys():
            segs = self.road_infor[road]
            road_st = generator.choice(busy_status, p=busy_ps[busy])
            segment, avg_t_r, avg_speed, intersection_l= create_segment(road_st, segs, generator)
            road_infor_change[road] = (segment, avg_t_r, avg_speed)
            all_intersections = all_intersections.union(intersection_l)
            segments += segment 
        Segment.segID = 0
        return segments, all_intersections, road_infor_change
        