from .decas import *
from .graph import *
from configs.systemcfg import *
from .onx_map import *
import time
class Road:
    """
        A class to represent a road in a map.
        Attributes
        ----------
        roadID : int
            A unique identifier for the road.
        rlong : list
            A list containing the minimum and maximum length of the road in kilometers.
        __rid : int
            The unique identifier for the road instance.
        __rinter : list
            A list of intersections associated with the road.
        generator : numpy.random.Generator
            A random number generator for generating random values.
        __rlong : int
            The length of the road in kilometers.
        __line : Line
            The line object representing the road.
        __state : int
            The current state of the road (e.g., normal, busy, etc.).
        __segments : list
            A list of segments that make up the road.
    """
    roadID = 0
    rlong = [1,5] #km
    def __init__(self, line, status = 0):
        self.__rid = Road.roadID
        Road.roadID+=1
        self.__rinter = []
        self.generator = np.random.default_rng(42)
        self.__rlong = self.generator.integers(Road.rlong[0], Road.rlong[1])
        self.__line = line
        self.__state = status
        self.__segments = []
    
    def update_id(self,val):
        self.__rid = val
    
    def __str__(self):
        pnt, k = self.__line.get_line()
        x, y = pnt.get_point()
        s=str(self.__rid )+','+str(x)+','+str(y) +','+ str(k)
        return s
    def get_avg_spped(self):
        return self.__avg_speed
    
    def get_task_rate(self):
        return self.__t_r
        
    def get_road(self):
        return (self.__rid, self.__rinter, self.__rlong)
    
    def update_rinter(self, val):
        if val in self.__rinter:
            return
        self.__rinter.append(val)
        
    def get_rinter(self):
        return self.__rinter
    
    def get_line(self):
        return self.__line
    
    def get_rid(self):
        return self.__rid
    
    def __eq__(self, __value):
        return self.__rid==__value.get_rid()
    
    def intersection_point(self, rd, max_v, max_h):
        return self.__line.intersection_point(rd.get_line(), max_v, max_h)
    
    def find_segments(self):
        """
        Divides the road into smaller segments based on the current state and 
        calculates the average speed and task rate for each segment.
        The method performs the following steps:
        1. Sorts the road intersections.
        2. Iterates through the sorted intersections to create segments.
        3. Depending on the current state, assigns different average speeds, 
           task rates, and statuses to each segment.
        4. Appends the created segments to the segments list.
        5. Calculates the mean task rate and average speed for all segments.
        Attributes:
            __rinter (list): List of road intersections.
            __state (int): Current state of the road.
            generator (object): Random generator for selecting statuses.
            __line (object): Line object representing the road.
            __segments (list): List to store the created segments.
            __t_r (float): Mean task rate for all segments.
            __avg_speed (float): Mean average speed for all segments.
        """
        
        #divide the road into smaller segments
        rinter = sorted(self.__rinter.copy())
        t_rs = []
        avg_speeds = []
        for i in range(len(rinter)-1):
            # if not self.__line.in_line(rinter[i]) or not self.__line.in_line(rinter[i+1]):
            #     continue
            if self.__state == 0 and rinter[i] != rinter[i+1]: 
                avg_speed = 20 #m/s
                t_r = 20 #tasks/s      
                status = 0 
                t_rs.append(t_r)     
                avg_speeds.append(avg_speed)     
            elif (self.__state == 1) and rinter[i] != rinter[i+1]:
                #set a random select status for the segment with a higet priority for status 0:                
                p = [0.95, 0.05]
                status = self.generator.choice([0,1], p= p)
                avg_speed = 17 #m/s
                t_r = 30 #tasks/s
                t_rs.append(t_r)     
                avg_speeds.append(avg_speed)   
            elif (self.__state == 2) and rinter[i] != rinter[i+1]:
                avg_speed = 14 #m/h
                t_r = 40 #tasks/s
                p = [0.7, 0.25, 0.05]
                status = self.generator.choice([0,1,2], p= p)
                t_rs.append(t_r)     
                avg_speeds.append(avg_speed)   
            elif (self.__state == 3) and rinter[i] != rinter[i+1]:
                avg_speed = 10 #m/s
                t_r = 50 #tasks/s
                p = [0.5, 0.25, 0.2, 0.05]
                status = self.generator.choice([0,1,2,3], p= p)
                t_rs.append(t_r)     
                avg_speeds.append(avg_speed)   
            elif (self.__state == 4) and rinter[i] != rinter[i+1]:
                avg_speed = 5 #m/s
                t_r = 60 #tasks/s
                p = [0.2, 0.2, 0.2, 0.2, 0.2]
                status = self.generator.choice([0,1,2,3, 4], p= p)
                t_rs.append(t_r)     
                avg_speeds.append(avg_speed)   
            else:
                continue
            self.__segments.append(Segment(rinter[i], rinter[i+1], status, self.__line, t_r, avg_speed))
        self.__t_r = np.mean(t_rs)
        self.__avg_speed = np.mean(avg_speeds)
    def get_segments(self):
        return self.__segments
    
    def get_infor(self):
        return 


class Map:
    """
    Represents a city map with roads, intersections, and segments, supporting both synthetic and real map generation.
    Attributes:
        busy_ps (list): Probability distributions for different traffic states.
        busy_status (list): List of traffic state identifiers.
        __rnum (int): Total number of roads in the map.
        __vrnum (int): Number of vertical roads.
        __hrnum (int): Number of horizontal roads.
        __busy (int): Current traffic state of the map.
        __max_v (int): Maximum vertical coordinate (map height).
        __max_h (int): Maximum horizontal coordinate (map width).
        __intersec (dict): Dictionary mapping intersection points to lists of roads.
        generator (np.random.Generator): Random number generator for reproducibility.
        edges (list): List of boundary roads.
        roads (list): List of all roads in the map.
        __segments (list): List of road segments.
        draw_d (dict): Dictionary for drawing segments by traffic state.
        hmap (HMap): Real map object (if applicable).
        __road_infor (Any): Information about roads in real map mode.
        __intersection_l (list): List of intersection points.
        hmap_updated (bool): Flag indicating if the real map has been updated.
        distance_dict (dict): Dictionary of shortest distances between segment endpoints.
    Methods:
        __init__(rnum, max_v=5000, max_h=5000, busy=0, fromfile=0):
            Initializes the Map object, generating roads and segments.
        load_map():
            Loads map configuration from a file and initializes roads.
        make_map():
            Generates a synthetic map with horizontal and vertical roads.
        build():
            Builds the map by finding intersections and segments, or loads a real map.
        buid_distance_map():
            Builds a dictionary of shortest distances between segment endpoints using Dijkstra's algorithm.
        update_hmap():
            Updates the real map's segments and intersections if not already updated.
        save_map():
            Saves the current map configuration to a file.
        update_intersec_road():
            Updates the intersection dictionary by finding intersections between all roads and edges.
        get_intersections():
            Returns the list of intersection points.
        find_path():
            Placeholder for a graph algorithm to find all possible routes between two points.
        draw_map():
            Draws the map using matplotlib, either synthetic or real map mode.
        get_segments():
            Returns the list of road segments.
        draw_segments():
            Draws the road segments with color coding based on traffic state.
        check_valid_nxt_intersection(current_pnt, visited):
            Returns the next valid intersection points for a vehicle, excluding visited points.
        test_graph():
            Tests the graph by finding possible routes between two intersection points using Dijkstra's algorithm.
    """
    
    # busy_status = ['None', "normal_time", "peak_time", "extem_crowed"]
    busy_ps =  [['1.0', "0.0", "0.0", "0.0","0.0"],
                ['0.7', "0.2", "0.1", "0.0", "0.0"],#None busy
                ['0.4', "0.3", "0.1", "0.1", "0.1"], #normal_time
                ['0.1', "0.1", "0.2", "0.3", "0.3"], #normal_time
                ['0.0', "0.1", "0.1", "0.3", "0.5"] #normal_time
                ]
    busy_status = [0,1,2,3,4]
    def __init__(self, rnum, max_v=5000, max_h=5000, busy = 0, fromfile = 0):
        #rnum: the number of road in map
        self.__rnum = rnum
        #vrnum: the number of road in the vezrontal
        self.__vrnum = rnum//2
        #hrnum: the number of road in the horizontal
        self.__hrnum = rnum-self.__vrnum
        #buse: the state of map, from this value we can set the state for road
        self.__busy = busy
        self.__max_v = max_v
        self.__max_h = max_h
        
        self.__intersec = {} #point and list of lines
        self.generator = np.random.default_rng(48)
        
        self.edges = []
        self.edges.append(Road(Line(pnt1=Point(0,0), k=np.tan(90*np.pi/180))))
        self.edges.append(Road(Line(pnt1=Point(self.__max_v,self.__max_h), k=np.tan(90*np.pi/180))))   
        self.edges.append(Road(Line(pnt1=Point(0,self.__max_h), k=np.tan(0))))   
        self.edges.append(Road(Line(pnt1=Point(0,0), k=np.tan(0))))
        if fromfile==False:
            self.make_map()
        elif not map_cfg['real_map']:
            self.load_map()
        self.hmap_updated = False
        self.build()
        # self.buid_distance_map()
    
    def load_map(self):
        fl = open("./src/physic_definition/system_base/map_infor.txt", 'r')
        lines = fl.readlines()
        self.roads = []
        for l in lines:
            l = l.strip()
            road_st = self.generator.choice(Map.busy_status, p=Map.busy_ps[self.__busy])
            params = l.split(',')
            rd = Line(pnt1=Point(float(params[1]),float(params[2])), k=float(params[3]))
            rd = Road(rd, road_st)
            rd.update_id(int(params[0]))
            self.roads.append(rd)
            
    
    def make_map(self):
        self.roads = []
        
        #create horizontal road
        x = 0
        y = 0
        # create a base road 
        hbase = Line(pnt1=Point(x,y), k=np.tan(85*np.pi/180))       
        self.roads.append(Road(hbase))
        for i in range(self.__hrnum-1):
            ang = self.generator.uniform(-5,5)
            dis = self.generator.integers(self.__max_h*0.7/self.__rnum, 3*self.__max_h/self.__rnum)
            x += dis
            hbase = hbase.get_line_from_angle(pnt=Point(x,y), angle=ang)
            road_st = self.generator.choice(Map.busy_status, p=Map.busy_ps[self.__busy])
            self.roads.append(Road(hbase, road_st))
            
        x = 0
        y = 0
        vbase = Line(pnt1=Point(x,y), k=np.tan(1*np.pi/180)) 
        self.roads.append(Road(vbase))
        for i in range(self.__vrnum-1):
            ang = self.generator.uniform(-5,5)
            dis = self.generator.integers(self.__max_h*0.7/self.__rnum, 3*self.__max_h/self.__rnum)
            y += dis
            vbase = vbase.get_line_from_angle(pnt=Point(x,y), angle=ang)
            road_st = self.generator.choice(Map.busy_status, p=Map.busy_ps[self.__busy])
            self.roads.append(Road(vbase,road_st))
            
    def build(self):
        if not map_cfg['real_map']:
            #call update de intersection
            self.draw_d = {0:[], 1: [], 2: [], 3:[], 4:[]}
            self.update_intersec_road()
            self.__segments = []
            Segment.segID = 0
            for rd in self.roads:
                rd.find_segments()
                sgms = rd.get_segments()
                self.__segments += sgms
                for sgm in sgms:
                    self.draw_d[sgm.get_state()].append(sgm.get_points())
        else:
            self.hmap = HMap(map_cfg['real_center_point'],map_cfg['radius'])
            seg_, road_infor_change, all_intersections = self.hmap.build_map(Map.busy_status,Map.busy_ps, self.__busy, self.generator)
            self.__segments = seg_
            self.__road_infor = road_infor_change
            self.__intersection_l = list(all_intersections)
            self.hmap_updated = True
    
    def buid_distance_map(self):
        gr = Graph(self.__segments)
        self.distance_dict = {}
        
        has_road = []
        
        print(len(self.__segments))
        for seg in self.__segments:
            pnts = seg.get_endpoints()
            for seg_ in self.__segments:
                pnts_ = seg.get_endpoints()
                for pnt in pnts:
                    for pnt_ in pnts_:
                        if (pnt,pnt_) not in has_road:
                            s = time.perf_counter()
                            gr.dijkstra(pnt, pnt_)
                            self.distance_dict[(pnt,pnt_)] = gr.get_min_long()
                            has_road.append((pnt,pnt_))
                            e = time.perf_counter()
                            print("time = {}".format(e-s))
                        
                
                          
    
    def update_hmap(self):
        if self.hmap_updated:
            return
        segments, all_intersections, road_infor_change = self.hmap.update_map(self.generator, Map.busy_status, Map.busy_ps, self.__busy)
        self.__segments = segments
        self.__intersection_l = list(all_intersections)
        self.__road_infor = road_infor_change
        self.hmap_updated = True
        
        
    def save_map(self):
        fl = open("map_infor.txt", 'w')
        for rd in self.roads:
            fl.write(str(rd))
            fl.write('\n')
        fl.close()
    def update_intersec_road(self):
        
        all_ckpoint = self.roads + self.edges
        
        self.__intersection_l = []
        for rd in all_ckpoint:
            for other_rd in all_ckpoint:
                #traverse all other and find the intersection
                if other_rd!=rd:
                    intersec = rd.intersection_point(other_rd, self.__max_v, self.__max_h)
                    self.__intersection_l.append(intersec)
                    if (intersec!=Point(float('inf'), float('inf'))):
                        rd.update_rinter(intersec)
                        # print(self.__intersec.keys())
                        if intersec in self.__intersec.keys():
                            if rd not in self.__intersec[intersec]:
                                self.__intersec[intersec].append(rd)
                            if other_rd not in self.__intersec[intersec]:
                                self.__intersec[intersec].append(other_rd)
                        else:
                            self.__intersec[intersec] = [rd]
                            self.__intersec[intersec].append(other_rd)
    
    def get_intersections(self):
        return self.__intersection_l
    
    def find_path(self):
        '''
            this function using graph algorithm to find all possible root btw two points.
        '''                        
           
    def draw_map(self):
        if not map_cfg['real_map']:
            for i in range(len(self.roads)):
                x1, y1 = self.roads[i].get_line().get_points(5000)
                plt.plot(x1, y1, label = "l" + str(i))
            plt.ylim(0,5000)
            plt.xlim(0,5000)
            plt.savefig('./current_map.png', dpi =300)
            plt.close()
            
        else:
            custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
            # set a radius of 500
            G = ox.graph_from_point(map_cfg['real_center_point'],map_cfg['radius'], custom_filter=custom_filter, network_type='drive')
            edge_colors = []
            for u, v, k in G.edges(keys=True):
                edge_colors.append('blue')
            ox.plot_graph(G, edge_color=edge_colors, edge_linewidth=2, node_size=0)
            plt.savefig("./current_map.png",dpi=300)
            plt.close()
            
    def get_segments(self):
        return self.__segments
    
    def draw_segments(self):
        if not map_cfg['real_map']:
            fig, ax = plt.subplots(figsize=(10, 10))
            for key in self.draw_d.keys():
                draw_info = self.draw_d[key]
                for tup in draw_info:
                    if key == 0:  # Trạng thái 1 - Đường Thông Thoáng
                        plt.plot(tup[0], tup[1], color='#0000CC', linewidth=2) 
                    elif key == 1:  # Trạng thái 2 - Giao Thông Bình Thường
                        plt.plot(tup[0], tup[1], color='#33CC33', linewidth=2)
                    elif key == 2:  # Trạng thái 3 - Giao Thông Trung Bình
                        plt.plot(tup[0], tup[1], color='#FFFF00', linewidth=2)
                    elif key == 3:  # Trạng thái 4 - Giao Thông Ùn Ứ
                        plt.plot(tup[0], tup[1], color='#FFA500', linewidth=2) 
                    elif key == 4:  # Trạng thái 5 - Tắc Nghẽn Nghiêm Trọng
                        plt.plot(tup[0], tup[1], color='#FF0000', linewidth=2)
            plt.ylim(0,self.__max_h)
            plt.xlim(0,self.__max_v)
            ax.set_title("City Map from Segments")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.savefig('./current_map_seg' +str(self.__busy)+ '.png', dpi =300)
            plt.close()
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            for segment in self.__segments:
                st_pnt, ed_pnt = segment.get_endpoints()  
                x_vals = [st_pnt.get_point()[0], ed_pnt.get_point()[0]] 
                y_vals = [st_pnt.get_point()[1], ed_pnt.get_point()[1]] 
                if segment.get_state() == 0:
                    color = '#0000CC'
                elif segment.get_state() == 1:
                    color = '#33CC33'
                elif segment.get_state() == 2:
                    color = '#FFFF00'
                elif segment.get_state() == 3:
                    color = '#FFA500'
                else:
                    color = '#FF0000'
                ax.plot(x_vals, y_vals, color=color, linewidth=2)
            ax.set_title("City Map from Segments")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.savefig('./current_map_seg' +str(self.__busy)+ '.png', dpi =300)
            plt.close()
    def check_valid_nxt_intersection(self, current_pnt, visited):
        #when a vehicle moving on the road it shoube know what is the next point he can go
        #then vehicle can select one of them
        rds = self.__intersec[current_pnt]
        next_valid_point = []
        for rd in rds:
            intersecs = rd.get_rinter()
            min_dis = float('inf')
            next_vlid = intersecs[0]
            for its in intersecs:
                if its not in visited and its.get_dis_to_point(current_pnt) < min_dis:
                    min_dis = its.get_dis_to_point(current_pnt)
                    next_vlid = its
            next_valid_point.append(next_vlid)
        return next_valid_point
    def test_graph(self):
        gr = Graph(self.__segments)
        vertexes = [*self.__intersec]
        possible_roots = gr.dijkstra(vertexes[6], vertexes[8])
        # print('DONE')
        return possible_roots