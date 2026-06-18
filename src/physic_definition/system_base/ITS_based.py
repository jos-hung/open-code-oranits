import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')) 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) 
sys.path.append(".")
from physic_definition.map.decas import Point, Line, Segment
import matplotlib.pyplot as plt
from physic_definition.map.graph import Graph
import json
import abc
from configs.systemcfg import mission_cfg, task_cfg, map_cfg
from physic_definition.map.map import *
from physic_definition.network.rate import *
import time
import threading

'''
the below is observer pattern to notify the finish task
'''
class Observer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, mission):
        #when the mission is finished the update will notify for the whole cars
        #that a mission already finish, then the cars check that there task ready to process or not
        pass
    
class Subject(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def register_observer(self, observer_obj):
        pass
    @abc.abstractmethod
    def remove_observer(self, observer_obj):
        pass
    @abc.abstractmethod
    def notify_observer(self):
        pass
    

class Mission(Subject):
    """A class to represent a mission in the ITS-based system."""    
    missionID = 0
    status = ["created", "inprocess", "done"]
    def __init__(self, dpart, desti, tslot, graph, verbose=False):
        self.__taskid = Mission.missionID # the task id of the mission
        self.__status = 0 #the status
        self.__dpart = dpart
        self.__desti = desti
        self.__tslot = tslot
        Mission.missionID += 1
        self.__depend_l = []
        
        self.__obervers = []
        self.__graph = graph
        
        self.__best_road = self.__graph.dijkstra(self.__dpart, self.__desti)
        self.__long = self.__graph.get_min_long()
        
        self.__profit = 50 #temp fix, we can change later
        self.verbose = verbose
    
    def get_mission_destination(self):
        return self.__desti
    
    def set_mid(self, val):
        self.__taskid = val
    def get_infor_to_mission(self,p):
        #từ 1 vị trí đến vị trí bắt đầu nhiệm vụ
        best_road = self.__graph.dijkstra(p, self.__dpart)
        #giá trị min long thay đổi sau mỗi lần chạy dijkstra.
        distance = self.__graph.get_min_long()[0]
        return best_road, distance
    
    def reset(self):
        Mission.missionID = 0
    
    def get_long(self):
        return self.__long
    
    def __lt__(self, other):
        return self.__long[0] < other.get_long()[0]
    
    def __eq__(self, other):
        if isinstance(other, int) or isinstance(other, np.int64):
            return other == self.__taskid
        if isinstance(other, Mission):
            return other.get_mid() == self.__taskid
        if isinstance(other, Point):
            return other == self.__desti
            
    def get_profit(self):
        return self.__profit
    
    def set_profit(self, v):
        self.__profit = v
    
    def update_profit(self, v):
        self.__profit += v
        
    def get_best_road(self):
        return self.__best_road
    
    def in_other_road(self,mission):
        #this function help us to know that the current mission can be in a road of another mission or not.
        other_best_road = mission.get_best_road()
        for idx, p in enumerate(self.__best_road):
            if(idx > len(other_best_road)-1):
                return False
            if self.__best_road[idx]!=other_best_road[idx]:
                return False
        return True
    
    def get_mid(self):
        return self.__taskid
    
    #notify function
    def register_observer(self, observer_obj):
        if isinstance(observer_obj, list):
            for obs in observer_obj:
                if obs not in self.__obervers:
                    self.__obervers.append(obs)
        else:
            if observer_obj not in self.__obervers:
                self.__obervers.append(observer_obj)
    
    def remove_observer(self, observer_obj):
        self.__obervers.remove(observer_obj)
        
    def set_observers(self, observer_list):
        if isinstance(observer_list, list):
            self.__obervers = observer_list
    
    def notify_observer(self, time):
        n_remove_depends = 0
        for obs in self.__obervers:
            n_remove_depends += obs.update(self, time)
        return n_remove_depends
    def update_depend(self, id):
        self.__depend_l.append(id)
        if len(self.__depend_l) == 0:
            self.__status = 1
        else:
            self.__status = 0
    
    def set_depends(self, missions):
        self.__depend_l = missions 
        if len(self.__depend_l) ==0:
            self.__status = 1
    def remove_depend(self, id):
        self.__depend_l.remove(id)   
        if len(self.__depend_l) == 0:
            self.__status = 1
        else:
            self.__status = 0
    def get_depends(self):
        return self.__depend_l
    
    def update_status(self, value, missions = None, time = 0):
        """
        Updates the status of the mission and notifies dependent missions and observers.
        Args:
            value (int): The new status value to be set for the mission.
            missions (list, optional): A list of Mission objects that depend on this mission. Defaults to None.
            time (int, optional): The current time to be used for notifying observers. Defaults to 0.
        Returns:
            int: The number of dependencies removed and observers notified.
        Example:
            mission = Mission()
            dependent_mission = Mission()
            mission.update_status(1, [dependent_mission], 10)
        """
       
        
        self.__status = value
        n_remove_depends = 0
        if Mission.status[value] == "done" and missions!=None:
            if self.verbose:
                print("Nhiệm vụ {} bắt đầu thông báo cho các phương tiện khác rằng nó đã được hoàn thành".format(self.get_mid()))
            for mis in missions:
                if self.get_mid() in mis.get_depends():
                    mis.__depend_l.remove(self.get_mid())
                    n_remove_depends += 1
        if Mission.status[value] == "done":
            n_remove_depends += self.notify_observer(time)
            if self.verbose:
                print("Nhiệm vụ {} bắt đầu thông báo cho các phương tiện khác rằng nó đã được hoàn thành".format(self.get_mid()))
        return n_remove_depends
    
    def get_status(self):
        return self.__status
    
    def print_status(self):
        print(Mission.status[self.__status])
    
    def get_dpart(self):
        return self.__dpart
    
    def get_desti(self):
        return self.__dpart

    def get_trajectory(self):
        return (self.__dpart, self.__dpart)
    
    def get_tslot(self):
        return self.__tslot
    
class Vehicle(Observer):
    """
    A class to represent a vehicle in the ITS-based system.
    The class manages the vehicle's position, missions, and interactions with the map.
    It allows setting missions, processing them, and updating the vehicle's state based on the missions' status.
    It also provides methods to accept missions, check readiness, and process tasks.
    """
    
    id = 0
    def __init__(self, cpu_freqz, cur_pos, map, tau = 120, verbose=False, sts = 0, non_priority_orders=False):
        #sts: là giá trị thể hiện rằng liệu phương tiện có chọn task by task ko
        self.__cpu_freqz = cpu_freqz
        self.__cur_pos = cur_pos
        self.__missions = []
        self.__ready_mis = []
        self.__acceptance_mis = []
        self.__map = map
        self.__vehicle_prof = 0
        self.__late = 0
        self.__vid = Vehicle.id
        Vehicle.id += 1
        self.__ctrl_time = 0
        self.__tau = tau*60
        self.__n_completes = 0
        self.__profit = 0
        self.verbose = verbose
        self.sts = sts
        self.__intime = True
        self.__mec = random_bs_num(map.get_intersections())
        self.__non_priority_orders = non_priority_orders
        self.__order = []
        
    def get_vid(self):
        return self.__vid
    
    def get_pos(self):
        return self.__cur_pos
    
    def set_pos(self,pos):
        self.__cur_pos=pos    
    
    def reset(self):
        Vehicle.id = 0
    
    def set_mission(self, sol, missions=[], mtuple = False):
        """
        Sets the mission for the vehicle based on the provided solution and missions.
        Parameters:
        sol (list, np.ndarray, int, np.int64): The solution which can be a list, numpy array, or an integer.
        missions (list): A list of mission objects.
        mtuple (bool): A flag indicating if the solution is a tuple containing order and vehicle information.
        Returns:
        bool: Returns True if there are ready missions, False otherwise.
        Raises:
        ValueError: If the input solution type is incorrect.
        Notes:
        - If `sol` is a list or numpy array and `mtuple` is False, it iterates through the solution to find the vehicle ID and sets the mission.
        - If `sol` is an integer or numpy int64, it directly sets the mission based on the index.
        - If `sol` is a list or numpy array and `mtuple` is True, it processes the solution as a tuple containing order and vehicle information.
        - The method updates the mission status and verifies readiness based on the mission dependencies and priority orders.
        """
        if self.__intime ==False:
            return
        if (isinstance(sol, list) or isinstance(sol, np.ndarray)) and not mtuple:
            for idx, val in enumerate(sol):
                #val = vehicle id
                if val == self.__vid:
                    i = missions.index(int(idx))
                    mis = missions[i]
                    self.accept_mission(miss=mis)
                    if len(mis.get_depends()) == 0:
                        mis.update_status(1)
                    if self.__non_priority_orders and mis.get_status() == 1:
                        self.__ready_mis.append(mis)
                        self.__missions.remove(mis)
                    else:
                        self.verify_ready()
        elif isinstance(sol, int) or isinstance(sol, np.int64):
            i = missions.index(sol) #index of mission
            mis = missions[i]
            if len(mis.get_depends()) == 0:
                mis.update_status(1)
            self.accept_mission(miss=mis)
            if self.__non_priority_orders and mis.get_status() == 1:
                self.__ready_mis.append(mis)
                self.__missions.remove(mis)
            else:
                self.verify_ready()
            if len(self.__ready_mis) == 0:
                return False
            return True
        elif (isinstance(sol, list) or isinstance(sol, np.ndarray)) and mtuple:
            completed_set = []
            for idx, val in enumerate(sol):
                #val = order and vehicle
                if val[1] == self.__vid and val not in completed_set:
                    i = missions.index(int(idx))
                    mis = missions[i]
                    if len(mis.get_depends()) == 0:
                        mis.update_status(1)                     
                    if self.__non_priority_orders and mis.get_status() == 1:
                        self.__ready_mis.append(mis)
                        self.__missions.remove(mis)
                    else:
                        self.verify_ready()
                    self.accept_mission(miss=mis, order = val[0])
                    completed_set.append(val)
                
        
        else:
            raise ValueError("wrong inputs")
            
    def fit_order(self):
        """
        Sorts and processes orders and missions based on their status.
        This method first sorts the orders by their second element. It then iterates through the sorted orders and 
        matches them with missions. If a mission's status is 1 and it is ready, the mission is moved to the ready 
        missions list and removed from the missions and acceptance missions lists. If the mission is not ready, 
        it is added to the acceptance list. Finally, the missions and acceptance missions lists are updated with 
        the accepted missions.
        Returns:
            None
        """
        
        if len(self.__order) ==0:
            return
        self.__order.sort(key = lambda x:x[1])
        accept = []
        ready = True
        for order in self.__order:
            for mis in self.__missions:
                if mis.get_mid() == order[0]:
                    if mis.get_status() == 1 and ready:
                        self.__ready_mis.append(mis)
                        self.__missions.remove(mis)
                        self.__acceptance_mis.remove(mis)
                    else:
                        ready = False
                        accept.append(mis)             
        self.__missions= self.__acceptance_mis = accept
        self.__acceptance_mis = accept.copy()
            
    def accept_mission(self, miss, order = None):
        """
        Accepts a mission and optionally assigns an order to it.
        This method registers the current object as an observer of the mission,
        appends the mission to the list of missions, and optionally appends the 
        mission and its order to the list of orders.
        Args:
            miss (Mission): The mission to be accepted.
            order (Optional[Any]): The order associated with the mission. Defaults to None.
        Returns:
            None
        """
        
        if order != None:
            self.__order.append((miss.get_mid(), order)) 
        miss.register_observer(self)
        self.__missions.append(miss)
        self.__acceptance_mis.append(miss)
    
    def get_ctrl_time(self):
        return self.__ctrl_time
    
    def update(self, mission, time = 0):
        """
        Update the system state based on the given mission and time.
        Args:
            mission: The mission object that contains the details of the mission to be updated.
            time (int, optional): The current time. Defaults to 0.
        Returns:
            int: The number of dependencies removed after checking readiness.
        """
        
        is_depends = False
        if len(self.__ready_mis)==0 :
            is_depends = True
        
        n_remove_depends = self.check_ready(mission=mission)
        if self.__ctrl_time < time and is_depends:
            self.__ctrl_time = time
        return n_remove_depends
    def check_ready(self, mission):
        """
        Check and update the readiness of missions based on dependencies.
        This method checks the current list of missions and updates their status 
        based on the dependencies of the given mission. If a mission has no more 
        dependencies, its status is updated to ready. The method also handles 
        the removal of dependencies and updates the list of ready missions.
        Args:
            mission: The mission object whose dependencies are to be checked.
        Returns:
            int: The number of dependencies removed.
        """
        
        if len(self.__missions) == 0:
            return 0
        id = mission.get_mid()
        n_remove_depends = 0
        for mis in self.__missions:
            deps = mis.get_depends()
            if id in deps:
                mis.remove_depend(id)
                n_remove_depends += 1
                if self.verbose:
                    print("Xe ô tô {} đã cập nhật việc xoá dependence {} của nhiệm vụ {}".format(self.__vid,id, mis.get_mid()))
            if len(deps)==0:
                mis.update_status(1, mission,self.__ctrl_time)
            if mis.get_status() == 1 and self.__non_priority_orders:
                self.__ready_mis.append(mis)
                self.__missions.remove(mis)
        
        if len(self.__missions[0].get_depends())==0 and not self.__non_priority_orders:
            mis = self.__missions.pop(0)
            mis.update_status(1, mission, self.__ctrl_time)
            if mis not in self.__ready_mis:
                self.__ready_mis.append(mis)
        return n_remove_depends      
    def verify_ready(self):
        if len(self.__missions) < 1:
            return
        if len(self.__missions[0].get_depends()) ==0 and len(self.__ready_mis)==0 and not self.__non_priority_orders:
            self.__ready_mis.append(self.__missions[0])
            if self.__missions[0] in self.__acceptance_mis:
                self.__acceptance_mis.remove(self.__missions[0])
            self.__missions.remove(self.__missions[0])
            
    
    def inprocess(self):
        return len(self.__ready_mis)>0    
    
    def check_time(self):
        return self.__intime
    
    def process_mission(self, missions = None):
        #process priority is the shortest mission first
        #check event two mission in the same parth or not
        #we shoud take care the case that, is there manupulate the parth of two or three  task to reduce the fee
        if len(self.__ready_mis) ==0:
            return
        elif self.__intime == False:
            return
       
        # sorted(self.__ready_mis) #sort các nhiệm vụ theo quãng đường thực hiện
        
        cur_mis = self.__ready_mis.pop(0)
        
        
        if len(cur_mis.get_depends())!=0:
            raise ValueError("The mission is not ready to compute ...")
        
        
        #bước 1, từ vị trí hiện tại của vehicle tính thời gian đến vị trí của mission.
        #để tính được nó, chúng ta có 2 bước:
        # ->bước 1, tìm vị trí hiện tại củ vehicle.
        # ->bước 2, tìm vị trí hiện tại củ mission.
        # ->bước 3, gọi hàm tính toán quảng đường, sau đó tính thời gian di chuyển tới nhiệm vụ.
        veh_position = self.__cur_pos
        best_trajectrory_to_mission, distance_to_mission = cur_mis.get_infor_to_mission(veh_position)

        '''
        khi có các nhiệm vụ trên cùng 1 cung đường,
        chúng ta chỉ cần lấy cung đường dài nhất.
        tính tổng của các profit và gán cho cung đường đó.
        '''
        same_longest_road_mis = cur_mis
        max_point = -float('inf')
        prof = cur_mis.get_profit()
        
        end_points= [cur_mis.get_best_road()[-1]] #count the mission finished or not
        same_roads = [cur_mis]
        
        total_length = cur_mis.get_long()[0]
        for mis in self.__ready_mis:
            if mis != cur_mis:
                if cur_mis.in_other_road(mis):
                    if max_point < len(mis.get_best_road()):
                        same_longest_road_mis = mis
                        max_point = len(mis.get_best_road())
                    prof += mis.get_profit()
                    self.__ready_mis.remove(mis)
                    end_points.append(mis.get_best_road()[-1])
                    same_roads.append(mis)
                    total_length += mis.get_long()[0]
        same_longest_road_mis.set_profit(prof)
        trajectory = same_longest_road_mis.get_best_road()

        segments = self.__map.get_segments()
 
        completed_cnt = 0
        total_delay = distance_to_mission/10 #thời gian xe đến vị trí của nhiệm vụ, bỏ qua thời gian offloading time, tốc độ trung bình 10 m/s.
        if len(best_trajectrory_to_mission)>0:
            best_trajectrory_to_mission.remove(best_trajectrory_to_mission[-1]) #loaị bỏ điểm cuối cùng là điểm bắt đầu nhiệm vụ
        trajectory = best_trajectrory_to_mission + trajectory
        
        while len(trajectory) > 1:
            cur_point = trajectory.pop(0)
            points = (cur_point, trajectory[0])
            idx= segments.index(points)
            
            cur_seg = segments[idx]
            _, aver_speed = cur_seg.get_infor()
            #offloadingtask is a dict key = seg_id, val = list of offoadling task
            offload_task = cur_seg.get_offloading_tasks()
            offloading_delay = 0
            
            current_road_long = cur_point.get_dis_to_point(trajectory[0])
            on_road_time = current_road_long/aver_speed
            
            cur_offloading_delay = 0
            for offt in offload_task:
                '''
                Trong thực tế chúng ta sẽ không biết đc khoảng thời gian giữa hai lần offloading.
                vì thế dựa vào tỷ số lambda khi task xuất hiện chúng ta có thể tính được liệu rằng sau bn lâu sẽ có 1 offloading task xuất hiện.
                từ đó có thể tính được vị trí mà xe đang offloading:
                vị trí thời gian cũ + vị trí thời gian mới
                vị trí ở thời gian mới chính bằng khoảng tgian trung bình generate task* tốc độ
                đó chính là inter_arrival_time
                '''
                current_line_of_road = Line(pnt1=cur_point, pnt2=trajectory[0])
                #update position by time
                
                if cur_offloading_delay == 0:
                    new_points = cur_point
                else:
                    current_pointxy = cur_point.get_point()
                    increase = 1
                    if current_pointxy[0] > trajectory[0].get_point()[0]:
                        increase = -1
                    l = aver_speed*cur_offloading_delay
                    slope = current_line_of_road.get_slop()
                    new_x = current_pointxy[0] + increase*(l/(sqrt(l/(1+slope**2))))
                    new_y = current_line_of_road.calculate_point(new_x)
                    new_points = Point(x = new_x, y=new_y)
                    
                # rate and cpu computation capacity
                rate, cpu_freq = get_rate_and_mec_cpu(new_points, self.__mec)
                task_ifor = offt.get_task()
                #delay
                communtion_delay = task_ifor[0]*8000/rate
                computing_delay = task_ifor[1]/cpu_freq #ignore the queuing delay
            
                if self.verbose:
                    print(rate, cpu_freq, communtion_delay, computing_delay)
                same_longest_road_mis.update_profit(-task_cfg['cost_coefi'])
                cur_offloading_delay = communtion_delay+ computing_delay
                offloading_delay += cur_offloading_delay
                
            total_delay += offloading_delay + on_road_time
            
            while cur_point in end_points:
                end_points.remove(cur_point)
                if self.__ctrl_time + total_delay< self.__tau:
                    completed_cnt += 1
                else:
                    break
                idx = same_roads.index(cur_point)
                mis = same_roads[idx]
                if mis in self.__missions:
                    self.__missions.remove(mis)
                if mis in self.__ready_mis:
                    self.__ready_mis.remove(mis)
                n_remove_depends = mis.update_status(2, missions, time = self.__ctrl_time+total_delay)
                self.set_pos(mis.get_desti())
                if self.verbose:
                    print("Nhiệm vụ {} hoàn thành tính toán".format(mis.get_mid()))
                    
        self.__ctrl_time += total_delay
    
        while trajectory[0] in end_points:
            end_points.remove(trajectory[0])
            if self.__ctrl_time < self.__tau:
                completed_cnt += 1
            else:
                break
            idx = same_roads.index(trajectory[0])
            mis = same_roads[idx]
            if mis in self.__missions:
                self.__missions.remove(mis)
            if mis in self.__ready_mis:
                self.__ready_mis.remove(mis)
            n_remove_depends = mis.update_status(2, missions, time = self.__ctrl_time)
            if self.verbose:
                print("Nhiệm vụ {} hoàn thành tính toán".format(mis.get_mid()))

        if self.__ctrl_time > self.__tau:
            self.__vehicle_prof -= (len(self.__missions) + len(self.__ready_mis))*50
            self.__missions = []
            self.__ready_mis = []
            self.__intime = False
            if self.verbose:
                print("Het thoi gian thuc hien nhiem vu") 
            return
        
        #nếu end_points >0 nghĩa là có 1 vài nhiệm vụ chưa hoàn thành,
        #việc của chúng ta là trace và giảm total profit đi
        total_mis_prof = 0
        while len(end_points) > 0:
            p = end_points.pop(0)
            idx = self.__acceptance_mis.index(p)
            total_mis_prof += self.__acceptance_mis[idx].get_profit()
            self.__late += 1
            
        B_remain = same_longest_road_mis.get_profit()-total_mis_prof
        if (B_remain<=0):
            B_remain = 0
        
        profit = total_length*0.025 + B_remain 
        self.__profit += B_remain
        self.__n_completes += completed_cnt
                
        self.__vehicle_prof += total_length*0.025 + B_remain 
                
        if self.__ctrl_time > self.__tau:
            self.__missions += self.__ready_mis
            self.__vehicle_prof -= self.get_profits(self.__missions)
            self.__missions = []
            self.__ready_mis = []
            self.__intime = False
            if self.verbose:
                print("Het thoi gian thuc hien nhiem vu")
        return self.__vid, cur_mis.get_mid(), n_remove_depends, len(self.__missions), profit

    def clear_total_reward(self):
        self.__vehicle_prof = 0
        self.__n_completes = 0
        self.__profit = 0
    
    def get_profits(self, mission):
        profit = 0
        if isinstance(mission, list):
            for mis in mission:
                profit += mis.get_profit()
        elif isinstance(mission, Mission):
            profit += mis.get_profit()
        return profit

    def get_accepted_missions(self):
        return self.__missions        
            
    def get_vhicle_prof(self):
        return self.__vehicle_prof
    
    def get_earn_profit(self):
        return self.__profit
    
    def get_earn_completes(self):
        return self.__n_completes

class Task:
    id = 0
    def __init__(self, datasize, compsize, id = None):
        self.__ds = datasize
        self.__cs = compsize
        self.__id = Task.id
        if id!=None:
            self.__id = id
        Task.id += 1

    def get_datasize(self):
        return self.__ds
    
    def get_compsize(self):
        return self.__cs 

    def get_task(self):
        return (self.__ds, self.__cs, self.__id)
    
    def __lt__(self, other):
        return self.__ds < other.__ds and self.__cs < other.__cs
    
    def __eq__(self, other):
        return self.__id == other.__id    
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, Task)):
            return list(obj.get_task())
        if isinstance(obj, (list, Point)):
            return list(obj.get_point())
            
        return super(NpEncoder, self).default(obj)


 
class TaskGenerator:
    def __init__(self, tau, map,
                 mindatasize = 100, #kbytes
                 maxdatasize = 500, #kbytes
                 mincompsize = 1, #mcycles
                 maxcompsize = 3, #mcycles
                 ):
        self.__map = map
        self.__map.draw_segments()
        self.__map.draw_map()
        # self.__map.save_map()
        self.__tau = tau
        self.__mindatasize = mindatasize
        self.__maxdatasize = maxdatasize
        self.__mincompsize = mincompsize
        self.__maxcompsize = maxcompsize
        self.generator = np.random.default_rng(42)
        segments = self.__map.get_segments()
        self.graph = Graph(segments)
    def update_graph(self, segments):
        self.graph = Graph(segments)
        
    def gen_tasks(self):
        '''
            suppose that the vehicle run with maximum speed, it have maximum number of offloading tasks.
            With maximum speed we assum that the offloading data is higher with lower speed.
            
            The below is a way to calculate the real number of task with current simulation speed.
            
            In this simulation, the vehicle provide the current average speed, then we can use the below equation to get the number of offloading tasks and its list.
            numtasks= (realsimulation/mximum)*number_of_task_with_maximum_speeds
            tasks = tasks[0:numtasks]
        '''
        segments = self.__map.get_segments()
        f= open("./task/task_information.json", "w")
        f.write("[")
        f.close()
        for seg in segments:
            filld = {}
            sta_seg = seg.get_state()
            longs = seg.get_long()
            lam, speed = seg.get_infor()
            runtime = longs/(speed/((sta_seg+1)*0.2))
            numtasks = int(lam*runtime)
            tasks = {}
            filld['seg_id'] = seg.get_sid()
            filld['seg_status'] = sta_seg
            filld['seg_longs'] = longs
            filld['seg_real_speed'] = (speed/((sta_seg+1)*0.2))
            filld['seg_max_numtask'] = numtasks
            for t in range(numtasks):
                datasize = self.generator.integers(self.__mindatasize, self.__maxdatasize)
                compsize = self.generator.integers(self.__mincompsize, self.__maxcompsize)
                task = Task(datasize=datasize, compsize=compsize)
                tasks[t] = task
            filld['seg_tasksl'] = tasks
            json_object = json.dumps(filld, indent=5, cls=NpEncoder)
            
            with open("./task/task_information.json", "a") as outfile:
                outfile.write(json_object)
                if seg!=segments[-1]:
                    outfile.write(',\n')
        f= open("./task/task_information.json", "a")
        f.write("]")
        f.close()
        Task.id = 0
        
    def gen_mission_non_file(self, nummissions):
        t = time.perf_counter()
        if map_cfg['real_map']:
            self.__map.update_hmap()
        get_vertexes_p = list(set(self.graph.get_vertexes()))
        avoi_cyc_dep={}
        missions = []
        
        task_depends = 0
        for i in range(nummissions):        
            seg_dep = self.generator.choice(get_vertexes_p)
            seg_des = self.generator.choice(get_vertexes_p)
            # Attempt to find a valid path
            # using straight path to avoid using  dijkstra, when the number of vertex to large, then the time increasing
            straight_path = seg_dep.get_dis_to_point(seg_des)
            times = straight_path / task_cfg['max_speed']
            while times > task_cfg['tau'] * 60 and times < 50 / task_cfg['max_speed']:
                seg_dep = self.generator.choice(get_vertexes_p)
                seg_des = self.generator.choice(get_vertexes_p)
                straight_path = seg_dep.get_dis_to_point(seg_des)
                times = straight_path / task_cfg['max_speed']
            mem_de = []
            if self.generator.uniform() >0.4:
                nums_depends = self.generator.integers(1,3)
                for k in range(nums_depends):
                    val = self.generator.integers(0, nummissions)
                    while (val == i):
                        val = self.generator.integers(0, nummissions)
                    if (val not in mem_de):
                        if val in avoi_cyc_dep.keys():
                            if i not in avoi_cyc_dep[val]:
                                mem_de.append(val)
                        else:
                            mem_de.append(val)
                task_depends += 1
            avoi_cyc_dep[i] = mem_de 
            depends = mem_de
            profit = self.generator.integers(mission_cfg['benifits'][0], mission_cfg['benifits'][1])
            m = Mission(seg_dep, seg_des, tslot=0, graph=self.graph)
            m.set_depends(depends)
            m.set_profit(profit)
            m.set_mid(i)      
            missions.append(m)
        Mission.missionID = 0
        print("generator time {}".format(time.perf_counter()-t))
        print("-----------> depend = ", task_depends, task_depends/mission_cfg['n_mission'])
        
        return missions, self.graph, self.__map
    def gen_mission(self, nummissions, file='mission_information.json'):
        t = time.perf_counter()
        if map_cfg['real_map']:
            self.__map.update_hmap()
        get_vertexes_p = list(set(self.graph.get_vertexes()))
        # missions = [] 
        f = open("./task/"+file, "w")
        f.write("[")
        f.close()
        
        #because of no cycle dependence, if A depend on B, then B cannot depend on A
        #to avoid this condition during generation, we use a map to handle it.
        
        avoi_cyc_dep={}
        
        for i in range(nummissions):
            filld = {}
            filld['i'] = i            
            seg_dep = self.generator.choice(get_vertexes_p)
            seg_des = self.generator.choice(get_vertexes_p)

            # Attempt to find a valid path
            # using straight path to avoid using  dijkstra, when the number of vertex to large, then the time increasing
            straight_path = seg_dep.get_dis_to_point(seg_des)
            times = straight_path / task_cfg['max_speed']
            while times > task_cfg['tau'] * 60 and times < 50 / task_cfg['max_speed']:
                seg_dep = self.generator.choice(get_vertexes_p)
                seg_des = self.generator.choice(get_vertexes_p)
                straight_path = seg_dep.get_dis_to_point(seg_des)
                times = straight_path / task_cfg['max_speed']
                
            
            mem_de = []
            if self.generator.uniform() >0.4:
                nums_depends = self.generator.integers(1,3)
                for k in range(nums_depends):
                    val = self.generator.integers(0, nummissions)
                    while (val == i):
                        val = self.generator.integers(0, nummissions)
                    if (val not in mem_de):
                        if val in avoi_cyc_dep.keys():
                            if i not in avoi_cyc_dep[val]:
                                mem_de.append(val)
                        else:
                            mem_de.append(val)
            avoi_cyc_dep[i] = mem_de 
            depends = mem_de
            profit = self.generator.integers(mission_cfg['benifits'][0], mission_cfg['benifits'][1])
            filld["depart_p"] = seg_dep
            filld["depart_s"] = seg_des
            filld["depends"] = depends
            filld["profit"] = profit
            json_object = json.dumps(filld, indent=4, cls=NpEncoder)
          
            # missions.append(m)
            
            with open("./task/"+file, "a") as outfile:
                outfile.write(json_object)
                if i!=nummissions-1:
                    outfile.write(',\n')
        f = open("./task/"+file, "a")
        f.write("]")
        f.close()
        e = time.perf_counter()
        print("generated time {}".format(e-t))
    def real_update(self, mission_f_name = "mission_information.json"):
        self.__map.update_hmap()
        self.gen_tasks()
        self.gen_mission(mission_cfg['n_mission'], file = mission_f_name)
def generate(mission_f_name = "mission_information.json"):
    map = Map(map_cfg['n_lines'], busy=map_cfg['busy'], fromfile=map_cfg['fromfile'])
    tg = TaskGenerator(tau = task_cfg['tau'], map=map,
                       mincompsize=task_cfg['comp_size'][0],
                       maxcompsize=task_cfg['comp_size'][1],
                       mindatasize=task_cfg['comm_size'][0],
                       maxdatasize=task_cfg['comm_size'][1],
                       )
    tg.gen_tasks()
    tg.gen_mission(mission_cfg['n_mission'], file = mission_f_name)
if __name__ == "__main__":
    
    generate()
    
