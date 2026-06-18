from numpy import sqrt 
import numpy as np
import matplotlib.pyplot as plt
class Point:
    def __init__(self, x = 0, y =0):
        self.__x = x
        self.__y = y
        
    def get_point(self):
        return (self.__x, self.__y)
     
    def get_dis_to_point(self, pnt):
        x, y = pnt.get_point()
        return (sqrt((x-self.__x)**2 + (y-self.__y)**2))
    
    def slope_with_point(self, pnt):
        x2, y2 = pnt.get_point()
        if x2 == self.__x:
            return float('inf')  # đường thẳng song song vs trục y
        else:
            return (y2 - self.__y) / (x2 - self.__x)
    def __eq__(self, value):
        if type(value) is Point:
            x , y = value.get_point()
        elif type(value) is tuple:
            x , y = value
        else:
            raise ValueError("the unknown supported type value {} {}".format( type(value), value))
        return (x == self.__x and y==self.__y)
    
    def __str__(self):
        return str(self.__x) + ", " +str(self.__y)
    
    def __hash__(self):
        return hash((self.__x, self.__y))
    
    def __lt__(self, v):
        return self.__x <v.get_point()[0]
         
class Line:
    def __init__(self, pnt1, k =0, pnt2 = Point(), slop = True):
        if slop:
            if k< -1.6e+16:
                self.__k = -float('inf')
            elif k > 1.6e+16:
                self.__k = float('inf')
            else:
                self.__k = k
        else:
            self.__k = pnt1.slope_with_point(pnt2) #Hệ số góc
        self.__pnt = pnt1 #1 điểm thuộc đường thẳng (base point)
    def get_slop(self):
        return self.__k
    
    def in_line(self, point, eps = 1):
        x0, y0 = point.get_point()
        y = self.calculate_point(x0)
        if abs(y - y0) < eps:
            return True
        return False
    
    def get_line(self):
        return (self.__pnt, self.__k)
    
    def get_base_point(self):
        return self.__pnt
    
    def calculate_point(self, x):
        x0, y0 = self.__pnt.get_point()
        # Sử dụng phương trình y = k(x - x0) + y0 để tính y
        return self.__k * (x - x0) + y0
    
    def get_line_from_angle(self, angle, pnt = Point()):
        '''
            This function returns a line which across with current line an angle = angle.
        '''
        angle_rad = angle * (np.pi / 180)
        k2 = np.tan(np.arctan(self.__k) - angle_rad)
        if k2 < -1.6e+16:
            k2 = -float('inf')
        elif k2 > 1.6e+16:
            k2 = float('inf')
        return Line(pnt1 = pnt, k = k2)
    
    def find_point_at_distance(self, distance, direction="right"):
        angle = np.arctan(self.__k)

        if direction == "right":
            dx = np.cos(angle) * distance
            dy = np.sin(angle) * distance
        else: 
            dx = -np.cos(angle) * distance
            dy = -np.sin(angle) * distance

        x, y = self.__pnt.get_point()
        new_x = x + dx
        new_y = y+ dy
        return Point(new_x, new_y)
    
    
    def get_line_points(self, num_points=100):
        x_vals = []
        y_vals = []
        x, y = self.__pnt.get_point()
        
        if self.__k!=float('inf') and self.__k!=-float('inf') and self.__k >=0:
            for i in range(1, num_points):
                x_ = x + i
                y_ = self.calculate_point(x_)
                x_vals.append(x_)
                y_vals.append(y_)
        elif self.__k!=float('inf') and self.__k!=-float('inf')and self.__k <0:
            for i in range(-num_points, 0):
                x_ = x + i
                y_ = self.calculate_point(x_)
                x_vals.append(x_)
                y_vals.append(y_)
        elif self.__k!=-float('inf'):
            for i in range(1, num_points):
                y_ = y + i
                x_vals.append(x)
                y_vals.append(y_)
        elif self.__k!=float('inf'):
            for i in range(-num_points, 0):
                y_ = y + i
                x_vals.append(x)
                y_vals.append(y_) 
        return x_vals, y_vals
    
    def get_points(self, num_points=100):
        x_vals = []
        y_vals = []
        x, y = self.__pnt.get_point()
        for i in range(1, num_points):
            x_ = x + i
            y_ = self.calculate_point(x_)
            x_vals.append(x_)
            y_vals.append(y_)
        return x_vals, y_vals

    def intersection_point(self, other_line, max_v, max_h):
        x1, y1 = self.__pnt.get_point()
        x2, y2 = other_line.get_base_point().get_point()
        # Giải phương trình hệ đồng dạng
        if self.__k == other_line.__k:
            # Đường thẳng song song, không có điểm giao nhau
            return Point(float('inf'), float('inf'))
        else:            
            if self.__k == float('inf'):
                # self là đường thẳng dựng đứng, tìm y trên other_line tại x = x1
                y_intersection = other_line.__k * (x1 - x2) + y2
                if (x1 < 0 or x1 > max_v or y_intersection < 0 or y_intersection > max_h):
                    return Point(float('inf'), float('inf'))
                return Point(x1, round(y_intersection,5))
            elif other_line.__k == float('inf'):
                # other_line là đường thẳng dựng đứng, tìm y trên self tại x = x2
                y_intersection = self.__k * (x2 - x1) + y1
                if (x2 < 0 or x2 > max_v or y_intersection < 0 or y_intersection > max_h):
                    return Point(float('inf'), float('inf'))
                return Point(x2, round(y_intersection,5))
            
            x_intersection = (y2 - y1 + self.__k * x1 - other_line.__k * x2) / (self.__k - other_line.__k)
            y_intersection = self.__k * (x_intersection - x1) + y1
            if (x_intersection  < 0 or x_intersection > max_v or y_intersection<0 or y_intersection >max_h):
                
                # print(x2, y2, other_line.__k, x_intersection, y_intersection, x_intersection  < 0, x_intersection > max_v, y_intersection >max_h)
                return Point(float('inf'), float('inf')) 
            return Point(round(x_intersection,5), round(y_intersection,5))

class Segment:
    segID = 0
    def __init__(self, st_pnt, ed_pnt, status, line=None, task_rate = None, avg_speed= None, length=None):
        self.__st_pnt = st_pnt
        self.__ed_pnt = ed_pnt
        self.__status = status
        self.__line = line
        self.__sid = Segment.segID
        if length!=None:
            self.__long = length
        else:
            self.__long = st_pnt.get_dis_to_point(ed_pnt)*(self.__status+1)
        self.__offloading_tasks = []
        self.__task_rate = task_rate
        self.__avg_speed = avg_speed
        Segment.segID +=1
    
    def get_infor(self):
        return (self.__task_rate, self.__avg_speed)
    def reset(self):
        Segment.segID = 0
    
    def set_offloading_tasks(self, v):
        self.__offloading_tasks = v
    
    def get_offloading_tasks(self):
        return self.__offloading_tasks
    
    def __str__(self):
        return str(self.__st_pnt) + "; " + str(self.__ed_pnt)
        
    def get_segment(self):
        return (self.__st_pnt, self.__ed_pnt, self.__line, self.__status, self.__sid)

    def get_endpoints(self):
        return (self.__st_pnt, self.__ed_pnt)
    
    def get_long(self):
        return self.__long
    
    def __eq__(self, other):
        if isinstance(other, Segment):
            return other.__sid == self.__sid
        if isinstance(other, tuple):
            return (other[0]== self.__st_pnt and other[1]== self.__ed_pnt) or \
                    (other[1]== self.__st_pnt and other[0]== self.__ed_pnt) 
        if isinstance(other, Point):
            #compare endpoint
            return self.__ed_pnt == other or self.__st_pnt == other
        if isinstance(other, int):
            #compare endpoint
            return self.__sid == other
    def get_points(self):
        x_vals = []
        y_vals = []
        x_st = self.__st_pnt.get_point()[0]
        x_ed = self.__ed_pnt.get_point()[0]
        while(x_st <x_ed):
            y_ = self.__line.calculate_point(x_st)
            x_vals.append(x_st)
            y_vals.append(y_)
            x_st +=  1
        return x_vals, y_vals
    
    def get_state(self):
        return self.__status
    
    def get_sid(self):
        return self.__sid
    
if __name__ == "__main__":
    # l6 = Line(pnt1 = Point(0,0), pnt2 = Point(1,0), slop=False)
    # l1 = Line(pnt1=Point(0,0), k=np.tan(85*np.pi/180)) 
    # l2 = l6.get_line_from_angle(-90, Point(0,0))
    # l3 = l2.get_line_from_angle(-45, Point(0,0))
    # l4 = l3.get_line_from_angle(-45, Point(0,0))
    # l5 = l4.get_line_from_angle(-45, Point(1,3))
    
    # x1, y1 = l1.get_line_points()
    # x2, y2 = l2.get_line_points()
    # x3, y3 = l3.get_line_points()
    # x4, y4 = l4.get_line_points()
    # x5, y5 = l5.get_line_points()
    
    # plt.plot(x1, y1, label = "l1")
    # plt.plot(x2, y2, label = "l2")
    # plt.plot(x3, y3, label = "l3")
    # plt.plot(x4, y4, label = "l4")
    # plt.plot(x5, y5, label = "l5")
    # plt.ylim(0,500)
    # plt.legend()
    # plt.show()
    
    # plt.close()
    
    x = 0
    y = 0
    roads = []
    hbase = Line(pnt1=Point(x,y), k=np.tan(85*np.pi/180)) 
    roads.append(hbase)
    for i in range(6):
        ang = np.random.uniform(-5, 5)
        dis = np.random.randint(700, 1500)
        x += dis
        hbase = hbase.get_line_from_angle(pnt=Point(x,y), angle=ang)
        roads.append(hbase)
        
    
    x = 0
    y = 40
    hbase = Line(pnt1=Point(x,y), k=np.tan(1*np.pi/180)) 
    roads.append(hbase)
    for i in range(5):
        ang = np.random.uniform(-5, 5)
        dis = np.random.randint(700, 1500)
        y += dis
        hbase = hbase.get_line_from_angle(pnt=Point(x,y), angle=ang)
        roads.append(hbase)    
    
    for i in range(len(roads)):
        x1, y1 = roads[i].get_points(5000)
        plt.plot(x1, y1, label = "l" + str(i))
    # plt.legend()
    plt.ylim(0,5000)
    plt.xlim(0,5000)
    plt.close()
    # plt.show()
