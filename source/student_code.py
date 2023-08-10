from expand import expand
import heapq
from collections import deque

visted = []
queue = []


import heapq

def a_star_search(dis_map, time_map, start, goal):
    if start not in dis_map or goal not in dis_map:
        return None  # Invalid start or end landmark
    
    open_list = [(0, start, [])]  # Priority queue contains tuples of (estimated total cost, current landmark, path)
    closed_set = set()

    while open_list:
        estimated_cost, current, path = heapq.heappop(open_list)
        
        if current == goal:
            return path + [current]
        
        if current not in closed_set:
            closed_set.add(current)
            for neighbor, distance in dis_map[current].items():
                if neighbor not in closed_set and distance is not None:
                    travel_time = time_map[current][neighbor]
                    if travel_time is not None:
                        new_path = path + [current]
                        heapq.heappush(open_list, (estimated_cost + distance, neighbor, new_path))

    return None  # No path found




def depth_first_search(time_map, start, end):
    if start not in time_map or end not in time_map:
        return None  # Invalid start or end landmark
    
    stack = [(start, [])]  # Stack contains tuples of (current landmark, path)
    visited = set()

    while stack:
        current, path = stack.pop()
        
        if current == end:
            return path + [current]
        
        if current not in visited:
            visited.add(current)
            for neighbor, travel_time in time_map[current].items():
                if neighbor not in visited and travel_time is not None:
                    stack.append((neighbor, path + [current]))

    return None  # No path found

def breadth_first_search(time_map, start, end):
    
    if start not in time_map or end not in time_map:
        return None  # Invalid start or end landmark
    
    queue = [(start, [])]  # Queue contains tuples of (current landmark, path)
    visited = set()

    while queue:
        current, path = queue.pop(0)
        
        if current == end:
            return path + [current]
        
        if current not in visited:
            visited.add(current)
            for neighbor, travel_time in time_map[current].items():
                if neighbor not in visited and travel_time is not None:
                    queue.append((neighbor, path + [current]))

    return None  # No path found


time_mapM = {
    'a': { 'a':None, 'b':1, 'c':None, 'd':None, 'e':1, 'f':None, 'g':None, 'h':None, 'i':None, 'j':None, 'k':None, 'l':None, 'm':None, 'n':None, 'o':None, 'p':None},
    'b': { 'a':1, 'b':None, 'c':1, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None, 'i':None, 'j':None, 'k':None, 'l':None, 'm':None, 'n':None, 'o':None, 'p':None},
    'c': { 'a':None, 'b':1, 'c':None, 'd':1, 'e':None, 'f':None, 'g':None, 'h':None, 'i':None, 'j':None, 'k':None, 'l':None, 'm':None, 'n':None, 'o':None, 'p':None},
    'd': { 'a':None, 'b':None, 'c':1, 'd':None, 'e':None, 'f':None, 'g':None, 'h':1, 'i':None, 'j':None, 'k':None, 'l':None, 'm':None, 'n':None, 'o':None, 'p':None},
    'e': { 'a':1, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None, 'i':1, 'j':None, 'k':None, 'l':None, 'm':None, 'n':None, 'o':None, 'p':None},
    'f': { 'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':1, 'h':None, 'i':None, 'j':1, 'k':None, 'l':None, 'm':None, 'n':None, 'o':None, 'p':None},
    'g': { 'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':1, 'g':None, 'h':1, 'i':None, 'j':None, 'k':None, 'l':None, 'm':None, 'n':None, 'o':None, 'p':None},
    'h': { 'a':None, 'b':None, 'c':None, 'd':1, 'e':None, 'f':None, 'g':1, 'h':None, 'i':None, 'j':None, 'k':None, 'l':None, 'm':None, 'n':None, 'o':None, 'p':None},
    'i': { 'a':None, 'b':None, 'c':None, 'd':None, 'e':1, 'f':None, 'g':None, 'h':None, 'i':None, 'j':None, 'k':None, 'l':None, 'm':1, 'n':None, 'o':None, 'p':None},
    'j': { 'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':1, 'g':None, 'h':None, 'i':None, 'j':None, 'k':None, 'l':None, 'm':None, 'n':1, 'o':None, 'p':None},
    'k': { 'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None, 'i':None, 'j':None, 'k':None, 'l':1, 'm':None, 'n':None, 'o':None, 'p':None},
    'l': { 'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None, 'i':None, 'j':None, 'k':1, 'l':None, 'm':None, 'n':None, 'o':None, 'p':1},
    'm': { 'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None, 'i':1, 'j':None, 'k':None, 'l':None, 'm':None, 'n':None, 'o':None, 'p':None},
    'n': { 'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None, 'i':None, 'j':1, 'k':None, 'l':None, 'm':None, 'n':None, 'o':1, 'p':None},
    'o': { 'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None, 'i':None, 'j':None, 'k':None, 'l':None, 'm':None, 'n':1, 'o':None, 'p':1},
    'p': { 'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None, 'i':None, 'j':None, 'k':None, 'l':1, 'm':None, 'n':None, 'o':1, 'p':None}
}

path = breadth_first_search(time_mapM, 'a', 'g')
print(path)






