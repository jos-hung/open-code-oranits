import numpy as np
from collections import deque
from .decas import *
import heapq

class Graph:
    def __init__(self, segs):
        self.__ngr = {} 
        self.__gr = {}  

        for s in segs:
            st, ed, *_ = s.get_segment()  
            self.__ngr.setdefault(st, []).append(s)
            self.__ngr.setdefault(ed, []).append(s) 
            self.__gr.setdefault(st, []).append(s)   

        print("finish build a graph from segment information")
    
    def get_vertexes(self, direction = False):
        if direction:
            return self.__gr.keys()
        return self.__ngr.keys()

    def dijkstra(self, pnt1: Point, pnt2: Point, directed: bool = False):
        if pnt1 not in self.__ngr or pnt2 not in self.__ngr:
            raise ValueError("Start or end point is not in the graph.")

        INF = float('inf')
        dist = {v: INF for v in self.__ngr.keys()}
        prev = {v: None for v in self.__ngr.keys()}
        hop = {}

        dist[pnt1] = 0.0
        hop[pnt1] = 0
        pq = [(0.0, 0, pnt1)]

        while pq:
            cur_dist, cur_hop, u = heapq.heappop(pq)
            if cur_dist != dist[u] or cur_hop != hop.get(u, cur_hop):
                continue
            if u == pnt2:
                break

            for ed in self.__ngr[u]:
                st, edp, line, status, sid = ed.get_segment()
                if u != st and u != edp:
                    continue
                neighbors = []
                if u == st:
                    neighbors.append(edp)        
                if not directed and u == edp:
                    neighbors.append(st)          
                w = ed.get_long()
                if w < 0:
                    raise ValueError(f"Negative weight not supported {w}")
                for v in neighbors:
                    alt = cur_dist + w
                    new_hop = cur_hop + 1
                    if alt < dist[v] or (alt == dist[v] and new_hop < hop.get(v, INF)):
                        dist[v] = alt
                        prev[v] = u
                        hop[v] = new_hop
                        heapq.heappush(pq, (alt, new_hop, v))

        if dist[pnt2] == INF:
            raise ValueError("No path exists between the given points.")

        # reconstruct
        path = []
        node = pnt2
        while node is not None:
            path.insert(0, node)
            node = prev[node]

        self.__minlong = (dist[pnt2], f"{pnt2.get_point()}")
        return path
    
    def get_min_long(self):
        return self.__minlong
    def get_graph(self, type = 'N'):
        if type =='N':
            return self.__ngr
        else:
            return self.__gr
        
    def get_possible_roots(self, ver_dp, ver_des, type='N'):
        #type N: undirect grap
        #type D: direct grap
        graph = self.__ngr if type == 'N' else self.__gr
        queue = deque()
        possible_roots = []

        # Start from all segments that begin with ver_dp
        if ver_dp in graph:
            for segment in graph[ver_dp]:
                queue.append((ver_dp, segment, [segment]))

        while queue:
            current_ver_dp, current_segment, path = queue.popleft()
            
            current_end = current_segment.get_segment()
            if current_ver_dp==current_end[0]:
                current_end=current_end[1]
            elif current_ver_dp==current_end[1]:
                current_end=current_end[0]
            
            # If we reach the destination, add the first segment of this path to possible_roots
            if current_end == ver_des and path not in possible_roots:
                possible_roots.append(path)
            
            for next_segment in graph.get(current_end, []):
                if next_segment not in path:  # Avoid cycles
                    queue.append((current_end ,next_segment, path + [next_segment]))
        return possible_roots
    
    def get_direct_possible_roots(self, ver_dp, ver_des, dir = "LR"):
        pass
    
    def get_undirect_possible_roots(self, ver_dp, ver_des):
        pass
    
        
        
        
        
    
                        
                        