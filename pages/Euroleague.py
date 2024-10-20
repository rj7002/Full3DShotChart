from euroleague_api.shot_data import ShotData
import io
import json
import numpy as np
import pandas as pd
import datetime
import requests
import plotly.graph_objs as go
# from abc import ABC, abstractmethod
import plotly.express as px
from datetime import datetime
from datetime import time
import time
from random import randint
import os
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns
# from matplotlib.patches import Polygon, Arc
# from scipy.spatial import ConvexHull
# import shapely
# import shapely.plotting
# import os
# from PIL import Image
import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.patches import Circle, Rectangle, Arc
# from scipy import ndimage
import streamlit as st
# import pandas as pd
# import numpy as np

# class CourtCoordinates:
#     '''
#     Stores court dimensions and calculates the (x,y,z) coordinates of the outside perimeter, 
#     three-point line, backboard, hoop, and free throw line.
#     '''
#     def __init__(self):
#         # New court dimensions to fit the expanded coordinate ranges
#         self.court_length = 700  # Expanded length to fit y range
#         self.court_width = 500   # Expanded width to fit x range
#         self.hoop_loc_x = 0      # Center of the court for expanded x range
#         self.hoop_loc_y = self.court_length / 2  # Center of the court for expanded y range
#         self.hoop_loc_z = 10    *10 # Height of the hoop
#         self.hoop_radius = .75*10
#         self.three_arc_distance = 23.75*10
#         self.three_straight_distance = 22*10
#         self.three_straight_length = 8.89*10
#         self.backboard_width = 6*10
#         self.backboard_height = 4*10
#         self.backboard_baseline_offset = 3*10
#         self.backboard_floor_offset = 9*10
#         self.free_throw_line_distance = 15*10

#     @staticmethod
#     def calculate_quadratic_values(a, b, c):
#         x1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
#         x2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
#         return x1, x2

#     def __get_court_perimeter_coordinates(self):
#         width = self.court_width
#         length = self.court_length
#         court_perimeter_bounds = [
#             [-width / 2, -length / 2, 0],
#             [width / 2, -length / 2, 0],
#             [width / 2, length / 2, 0],
#             [-width / 2, length / 2, 0],
#             [-width / 2, -length / 2, 0]
#         ]
#         court_df = pd.DataFrame(court_perimeter_bounds, columns=['x', 'y', 'z'])
#         court_df['line_group'] = 'outside_perimeter'
#         court_df['color'] = 'court'
#         return court_df

#     def __get_half_court_coordinates(self):
#         width = self.court_width
#         half_length = self.court_length / 2
#         circle_radius = 6 * 10
#         circle_radius2 = 2 * 10
#         circle_center = [0, half_length-600, 0]
#         circle_points = []
#         circle_points2 = []
#         num_points = 400
#         for i in range(num_points):
#             angle = 2 * np.pi * i / num_points
#             x = circle_center[0] + circle_radius * np.cos(angle)
#             y = circle_center[1] + circle_radius * np.sin(angle)
#             circle_points.append([x, y, circle_center[2]])
#         for i in range(num_points):
#             angle = 2 * np.pi * i / num_points
#             x = circle_center[0] + circle_radius2 * np.cos(angle)
#             y = circle_center[1] + circle_radius2 * np.sin(angle)
#             circle_points2.append([x, y, circle_center[2]])

#         half_court_bounds = [[(-width / 2), half_length-600, 0], [(width / 2), half_length-600, 0]]

#         half_df = pd.DataFrame(half_court_bounds, columns=['x', 'y', 'z'])
#         circle_df = pd.DataFrame(circle_points, columns=['x', 'y', 'z'])
#         circle_df['line_group'] = 'free_throw_circle'
#         circle_df['color'] = 'court'

#         circle_df2 = pd.DataFrame(circle_points2, columns=['x', 'y', 'z'])
#         circle_df2['line_group'] = 'free_throw_circle'
#         circle_df2['color'] = 'court'

#         half_df['line_group'] = 'half_court'
#         half_df['color'] = 'court'

#         return pd.concat([half_df, circle_df, circle_df2])

#     def __get_backboard_coordinates(self, loc):
#         backboard_start = -self.backboard_width / 2
#         backboard_end = self.backboard_width / 2
#         height = self.backboard_height
#         floor_offset = self.backboard_floor_offset

#         if loc == 'far':
#             offset = self.court_length / 2 - self.backboard_baseline_offset
#         elif loc == 'near':
#             offset = -self.court_length / 2 + self.backboard_baseline_offset

#         backboard_bounds = [
#             [backboard_start, offset, floor_offset],
#             [backboard_start, offset, floor_offset + height],
#             [backboard_end, offset, floor_offset + height],
#             [backboard_end, offset, floor_offset],
#             [backboard_start, offset, floor_offset]
#         ]

#         smaller_rect_width = 1.5*10
#         smaller_rect_height = 1*10
#         hoop_height = self.hoop_loc_z

#         smaller_rect_start_x = backboard_start + (self.backboard_width / 2) - (smaller_rect_width / 2)
#         smaller_rect_end_x = backboard_start + (self.backboard_width / 2) + (smaller_rect_width / 2)
#         smaller_rect_y = offset

#         smaller_rect_bounds = [
#             [smaller_rect_start_x, offset, hoop_height],
#             [smaller_rect_start_x, offset, hoop_height + smaller_rect_height],
#             [smaller_rect_end_x, offset, hoop_height + smaller_rect_height],
#             [smaller_rect_end_x, offset, hoop_height],
#             [smaller_rect_start_x, offset, hoop_height]
#         ]

#         backboard_df = pd.DataFrame(backboard_bounds, columns=['x', 'y', 'z'])
#         backboard_df['line_group'] = f'{loc}_backboard'
#         backboard_df['color'] = 'backboard'

#         smaller_rect_df = pd.DataFrame(smaller_rect_bounds, columns=['x', 'y', 'z'])
#         smaller_rect_df['line_group'] = f'{loc}_smaller_rectangle'
#         smaller_rect_df['color'] = 'backboard'

#         return pd.concat([backboard_df, smaller_rect_df])

#     def __get_three_point_coordinates(self, loc):
#         hoop_loc_x, hoop_loc_y = self.hoop_loc_x, self.hoop_loc_y
#         strt_dst_start = -self.court_width / 2 + self.three_straight_distance
#         strt_dst_end = self.court_width / 2 - self.three_straight_distance
#         strt_len = self.three_straight_length
#         arc_dst = self.three_arc_distance

#         start_straight = [
#             [strt_dst_start, -self.court_length / 2, 0],
#             [strt_dst_start, -self.court_length / 2 + strt_len, 0]
#         ]
#         end_straight = [
#             [strt_dst_end, -self.court_length / 2 + strt_len, 0],
#             [strt_dst_end, -self.court_length / 2, 0]
#         ]
#         line_coordinates = []

#         if loc == 'near':
#             hoop_loc_y = self.court_length / 2 - hoop_loc_y
#             start_straight = [[strt_dst_start, self.court_length / 2, 0], [strt_dst_start, self.court_length / 2 - strt_len, 0]]
#             end_straight = [[strt_dst_end, self.court_length / 2 - strt_len, 0], [strt_dst_end, self.court_length / 2, 0]]

#         a = 1
#         b = -2 * hoop_loc_y
#         d = arc_dst
#         for x_coord in np.linspace(int(strt_dst_start), int(strt_dst_end), 100):
#             c = hoop_loc_y ** 2 + (x_coord - hoop_loc_x) ** 2 - d ** 2

#             y1, y2 = self.calculate_quadratic_values(a, b, c)
#             if loc == 'far':
#                 y_coord = y1
#             if loc == 'near':
#                 y_coord = y2

#             line_coordinates.append([x_coord, y_coord, 0])

#         line_coordinates.extend(end_straight)

#         far_three_df = pd.DataFrame(line_coordinates, columns=['x', 'y', 'z'])
#         far_three_df['line_group'] = f'{loc}_three'
#         far_three_df['color'] = 'court'

#         return far_three_df

#     def __get_hoop_coordinates(self):
#     # Define the number of points to approximate the circle
#         num_points = 100
#         angle_step = 2 * np.pi / num_points

#         # Generate the circle points
#         angles = np.linspace(0, 2 * np.pi, num_points)
#         x = self.hoop_loc_x + self.hoop_radius * np.cos(angles)
#         y = self.hoop_loc_y - self.hoop_radius - 35 + self.hoop_radius * np.sin(angles)  # Adjust the center if needed
#         z = np.full_like(x, 100)  # Assuming a constant z value

#         # Create DataFrame
#         hoop_df = pd.DataFrame({
#             'x': x,
#             'y': y,
#             'z': z
#         })

#         # Add extra columns
#         hoop_df['line_group'] = 'hoop'
#         hoop_df['color'] = 'hoop'
        
#         return hoop_df

#     def get_coordinates(self):
#         court_perimeter_df = self.__get_court_perimeter_coordinates()
#         half_court_df = self.__get_half_court_coordinates()
#         far_backboard_df = self.__get_backboard_coordinates('far')
#         near_backboard_df = self.__get_backboard_coordinates('near')
#         far_three_df = self.__get_three_point_coordinates('far')
#         near_three_df = self.__get_three_point_coordinates('near')
#         hoop_df = self.__get_hoop_coordinates()

#         coordinates = pd.concat([
#             court_perimeter_df,
#             half_court_df,
#             far_backboard_df,
#             # near_backboard_df,
#             # far_three_df,
#             # near_three_df,
#             hoop_df
#         ])
        
#         return coordinates

import pandas as pd
import numpy as np

class CourtCoordinates:
    def __init__(self, year):
        # Court dimensions in cm
        self.court_length = 2833  # length of the court
        self.court_width = 1500    # width of the court
        self.hoop_height = 305     # height of the hoop
        self.hoop_loc_x = 0
        self.hoop_loc_y = 132       # adjusted to be within the court
        self.hoop_loc_z = self.hoop_height
        self.court_perimeter_coordinates = []
        self.three_point_line_coordinates = []
        self.backboard_coordinates = []
        self.hoop_coordinates = []
        self.free_throw_line_coordinates = []
        self.court_lines_coordinates_df = pd.DataFrame()
        self.year = year

    @staticmethod
    def calculate_quadratic_values(a, b, c):
        x1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        x2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        return x1, x2

    def calculate_court_perimeter_coordinates(self):
        # Calculate perimeter based on dimensions
        court_perimeter_bounds = [
    [-self.court_width / 2, -10, 0], 
    [self.court_width / 2, -10, 0], 
    [self.court_width / 2, self.court_length - 10, 0], 
    [-self.court_width / 2, self.court_length - 10, 0], 
    [-self.court_width / 2, -10, 0]
]
        self.court_perimeter_coordinates = court_perimeter_bounds

    def calculate_three_point_line_coordinates(self):
        # Three-point line radius for FIBA
        three_point_radius = 675  # radius in cm

        # line_coordinates = [[-three_point_radius, 0, 0], [-three_point_radius, 350, 0]]
        line_coordinates = []
        for x_coord in range(-three_point_radius+16, three_point_radius-20, 2):
            a = 1
            b = -2 * self.hoop_loc_y
            c = self.hoop_loc_y ** 2 + (self.hoop_loc_x - x_coord) ** 2 - (three_point_radius) ** 2
            y_coord = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
            line_coordinates.append([x_coord, y_coord, 0])

        line_coordinates.append([three_point_radius-15, 200, 0])
        line_coordinates.append([three_point_radius-15, 0, 0])

        # line_coordinates.append([-three_point_radius+15, 200, 0])
        # line_coordinates.append([-three_point_radius+15, 0, 0])

        self.three_point_line_coordinates = line_coordinates
    
    def addextraline(self):
        three_point_radius = 675  # radius in cm
        line_coordinates = []
        line_coordinates.append([-three_point_radius+15, 275, 0])
        line_coordinates.append([-three_point_radius+15, 0, 0])
        self.extralines = line_coordinates

    def calculate_backboard_coordinates(self):
        backboard_width = 180  # Backboard width in cm
        backboard_height = 305  # Backboard height in cm

        backboard_coordinates = [
            [backboard_width / 2, self.hoop_loc_y-25, 305], 
            [backboard_width / 2, self.hoop_loc_y-25, 395], 
            [-backboard_width / 2, self.hoop_loc_y-25, 395], 
            [-backboard_width / 2, self.hoop_loc_y-25, 305], 
            [backboard_width / 2, self.hoop_loc_y-25, 305]
        ]
        self.backboard_coordinates = backboard_coordinates

    def calculate_hoop_coordinates(self):
        hoop_radius = 45.72 / 2  # Rim radius in cm
        hoop_coordinates_top_half = []
        hoop_coordinates_bottom_half = []

        for hoop_coord_x in np.arange(-hoop_radius, hoop_radius + 0.5, 0.5):
            y_val = (hoop_radius ** 2 - hoop_coord_x ** 2) ** 0.5
            hoop_coordinates_top_half.append([hoop_coord_x, self.hoop_loc_y + y_val, self.hoop_loc_z])
            hoop_coordinates_bottom_half.append([hoop_coord_x, self.hoop_loc_y - y_val, self.hoop_loc_z])

        self.hoop_coordinates = hoop_coordinates_top_half + hoop_coordinates_bottom_half[::-1]

    def calculate_free_throw_line_coordinates(self):
        radius = 187.5  # Radius of free-throw arc in cm
        circle_center = [0, 500 - 25, 0]  # Adjusted center for size
        circle_points = []
        num_points = 100

        for i in range(num_points):
            angle = np.pi * i / (num_points - 1)
            x = circle_center[0] + radius * np.cos(angle)
            y = circle_center[1] + radius * np.sin(angle)
            circle_points.append([x, y, 0])
        
        free_throw_line_coordinates = [
            [-225, circle_center[1], 0],  # Left end of the line
            [225, circle_center[1], 0]    # Right end of the line
        ]

        baseline_y = 0
        lines_to_baseline = [
            [-225, circle_center[1], 0],
            [-225, baseline_y, 0],
            [225, circle_center[1], 0],
            [225, baseline_y, 0]
        ]

        self.free_throw_line_coordinates = circle_points + free_throw_line_coordinates + lines_to_baseline

    def __get_hoop_coordinates2(self):
        num_net_lines = 20
        net_length = 43.18  # Net length in cm
        initial_radius = 45.72 / 2  # Rim radius in cm

        hoop_net_coordinates = []
        for i in range(num_net_lines):
            angle = (i * 2 * np.pi) / num_net_lines
            
            for j in np.linspace(0, net_length, num=10):
                current_radius = initial_radius * (1 - (j / net_length) * 0.5)
                x = self.hoop_loc_x + current_radius * np.cos(angle)
                y = self.hoop_loc_y + current_radius * np.sin(angle)
                z = self.hoop_loc_z - j
                
                hoop_net_coordinates.append([x, y, z])
        
        self.hoop_net_coordinates = hoop_net_coordinates
    






    def calculate_three_point_line_coordinates2(self):
        # Three-point line radius for FIBA
        three_point_radius = 675  # radius in cm

        # line_coordinates = [[-three_point_radius, 0, 0], [-three_point_radius, 350, 0]]
        line_coordinates = []
        for x_coord in range(-three_point_radius + 16, three_point_radius - 20, 2):
            a = 1
            b = -2 * self.hoop_loc_y
            c = self.hoop_loc_y ** 2 + (self.hoop_loc_x - x_coord) ** 2 - (three_point_radius) ** 2
            y_coord = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
            
            # Flip the y coordinate
            flipped_y_coord = 2833 - y_coord
            line_coordinates.append([x_coord, flipped_y_coord, 0])

        line_coordinates.append([three_point_radius-15, 2833-200, 0])
        line_coordinates.append([three_point_radius-15, 2833-0, 0])

        self.three_point_line_coordinates = line_coordinates
    
    def addextraline2(self):
        three_point_radius = 675  # radius in cm
        line_coordinates = []
        line_coordinates.append([-three_point_radius+15, 2833-275, 0])
        line_coordinates.append([-three_point_radius+15, 2833-0, 0])
        self.extralines = line_coordinates

    def calculate_backboard_coordinates2(self):
        backboard_width = 180  # Backboard width in cm
        backboard_height = 305  # Backboard height in cm

        backboard_coordinates = [
            [backboard_width / 2, 2833-self.hoop_loc_y+25, 305], 
            [backboard_width / 2, 2833-self.hoop_loc_y+25, 395], 
            [-backboard_width / 2, 2833-self.hoop_loc_y+25, 395], 
            [-backboard_width / 2, 2833-self.hoop_loc_y+25, 305], 
            [backboard_width / 2, 2833-self.hoop_loc_y+25, 305]
        ]
        self.backboard_coordinates = backboard_coordinates

    def calculate_hoop_coordinates2(self):
        hoop_radius = 45.72 / 2  # Rim radius in cm
        hoop_coordinates_top_half = []
        hoop_coordinates_bottom_half = []

        for hoop_coord_x in np.arange(-hoop_radius, hoop_radius + 0.5, 0.5):
            y_val = (hoop_radius ** 2 - hoop_coord_x ** 2) ** 0.5
            hoop_coordinates_top_half.append([hoop_coord_x, 2833-self.hoop_loc_y + y_val, self.hoop_loc_z])
            hoop_coordinates_bottom_half.append([hoop_coord_x, 2833-self.hoop_loc_y - y_val, self.hoop_loc_z])

        self.hoop_coordinates = hoop_coordinates_top_half + hoop_coordinates_bottom_half[::-1]

    def calculate_free_throw_line_coordinates2(self):
        radius = 187.5  # Radius of free-throw arc in cm
        circle_center = [0, 2833 - 500 + 25, 0]  # Adjusted center for size
        circle_points = []
        num_points = 100

        for i in range(num_points):
            angle = np.pi + (np.pi * i / (num_points - 1))  # Flip the angle range to [π, 2π]
            x = circle_center[0] + radius * np.cos(angle)
            y = circle_center[1] + radius * np.sin(angle)
            circle_points.append([x, y, 0])
        
        free_throw_line_coordinates = [
            [-225, circle_center[1], 0],  # Left end of the line
            [225, circle_center[1], 0]    # Right end of the line
        ]

        baseline_y = 2833-0
        lines_to_baseline = [
            [-225, circle_center[1], 0],
            [-225, baseline_y, 0],
            [225, circle_center[1], 0],
            [225, baseline_y, 0]
        ]

        self.free_throw_line_coordinates = circle_points + free_throw_line_coordinates + lines_to_baseline

    def __get_hoop_coordinates4(self):
        num_net_lines = 20
        net_length = 43.18  # Net length in cm
        initial_radius = 45.72 / 2  # Rim radius in cm

        hoop_net_coordinates = []
        for i in range(num_net_lines):
            angle = (i * 2 * np.pi) / num_net_lines
            
            for j in np.linspace(0, net_length, num=10):
                current_radius = initial_radius * (1 - (j / net_length) * 0.5)
                x = self.hoop_loc_x + current_radius * np.cos(angle)
                y = 2833-self.hoop_loc_y + current_radius * np.sin(angle)
                z = self.hoop_loc_z - j
                
                hoop_net_coordinates.append([x, y, z])
        
        self.hoop_net_coordinates = hoop_net_coordinates

    def add_half_court_line(self):
        court_length = 2833  # Court length in cm
        half_court_x = 0     # X coordinate for the half-court line
        line_coordinates = []

        # Define the y-coordinates for the half-court line
        line_coordinates.append([-750, court_length/2, 0])  # Start point
        line_coordinates.append([750, court_length/2, 0])    # End point

        self.half_court_line = line_coordinates
    def add_half_court_circle(self):
        radius = 187.5  # Radius of the semicircle in cm
        circle_center = [0, 2833 / 2, 0]  # Center at the middle of the court height
        circle_points = []
        num_points = 100  # Number of points to create a smooth arc

        # Generate points for the top half of the circle
        for i in range(num_points // 2 + 1):  # Half points for the upper semicircle
            angle = np.pi * i / (num_points // 2)  # From 0 to π
            x = circle_center[0] + radius * np.cos(angle)
            y = circle_center[1] + radius * np.sin(angle)
            circle_points.append([x, y, 0])

        # Generate points for the bottom half of the circle
        for i in range(num_points // 2 + 1):  # Half points for the lower semicircle
            angle = np.pi * (1 - (i / (num_points // 2)))  # From π to 0
            x = circle_center[0] + radius * np.cos(angle)
            y = circle_center[1] - radius * np.sin(angle)
            circle_points.append([x, y, 0])

        self.half_court_circle = circle_points







    def calculate_court_lines_coordinates(self):

        self.add_half_court_circle()
        circledf = pd.DataFrame(self.half_court_circle, columns=['x', 'y', 'z'])
        circledf['line_id'] = 'halfcircle'
        circledf['line_group_id'] = 'circlecourt'

        self.add_half_court_line()
        halfdf = pd.DataFrame(self.half_court_line, columns=['x', 'y', 'z'])
        halfdf['line_id'] = 'half'
        halfdf['line_group_id'] = 'hcourt'

        self.calculate_court_perimeter_coordinates()
        court_df = pd.DataFrame(self.court_perimeter_coordinates, columns=['x', 'y', 'z'])
        court_df['line_id'] = 'outside_perimeter'
        court_df['line_group_id'] = 'court'

        self.calculate_three_point_line_coordinates()
        three_point_line_df = pd.DataFrame(self.three_point_line_coordinates, columns=['x', 'y', 'z'])
        three_point_line_df['line_id'] = 'three_point_line'
        three_point_line_df['line_group_id'] = 'threepoint'
        
        self.addextraline()
        extradf = pd.DataFrame(self.extralines, columns=['x', 'y', 'z'])
        extradf['line_id'] = 'three2'
        extradf['line_group_id'] = 'threepoint2'


        self.calculate_backboard_coordinates()
        backboard_df = pd.DataFrame(self.backboard_coordinates, columns=['x', 'y', 'z'])
        backboard_df['line_id'] = 'backboard'
        backboard_df['line_group_id'] = 'backboard'

        self.__get_hoop_coordinates2()
        netdf = pd.DataFrame(self.hoop_net_coordinates, columns=['x', 'y', 'z'])
        netdf['line_id'] = 'hoop2'
        netdf['line_group_id'] = 'hoop2'

        self.calculate_hoop_coordinates()
        hoop_df = pd.DataFrame(self.hoop_coordinates, columns=['x', 'y', 'z'])
        hoop_df['line_id'] = 'hoop'
        hoop_df['line_group_id'] = 'hoop'

        self.calculate_free_throw_line_coordinates()
        free_throw_line_df = pd.DataFrame(self.free_throw_line_coordinates, columns=['x', 'y', 'z'])
        free_throw_line_df['line_id'] = 'free_throw_line'
        free_throw_line_df['line_group_id'] = 'free_throw_line'

        self.court_lines_coordinates_df = pd.concat([
            court_df, three_point_line_df, backboard_df, hoop_df, free_throw_line_df, netdf,extradf
        ], ignore_index=True, axis=0)




        self.calculate_three_point_line_coordinates2()
        three_point_line_df2 = pd.DataFrame(self.three_point_line_coordinates, columns=['x', 'y', 'z'])
        three_point_line_df2['line_id'] = 'three_point_line2'
        three_point_line_df2['line_group_id'] = 'threepoint2.1'
        
        self.addextraline2()
        extradf2 = pd.DataFrame(self.extralines, columns=['x', 'y', 'z'])
        extradf2['line_id'] = 'three2.2'
        extradf2['line_group_id'] = 'threepoint2.2'


        self.calculate_backboard_coordinates2()
        backboard_df2 = pd.DataFrame(self.backboard_coordinates, columns=['x', 'y', 'z'])
        backboard_df2['line_id'] = 'backboard2'
        backboard_df2['line_group_id'] = 'backboard2'

        self.__get_hoop_coordinates4()
        netdf2 = pd.DataFrame(self.hoop_net_coordinates, columns=['x', 'y', 'z'])
        netdf2['line_id'] = 'hoop2.2'
        netdf2['line_group_id'] = 'hoop2.2'

        self.calculate_hoop_coordinates2()
        hoop_df2 = pd.DataFrame(self.hoop_coordinates, columns=['x', 'y', 'z'])
        hoop_df2['line_id'] = 'hoop2.1'
        hoop_df2['line_group_id'] = 'hoop2.1'

        self.calculate_free_throw_line_coordinates2()
        free_throw_line_df2 = pd.DataFrame(self.free_throw_line_coordinates, columns=['x', 'y', 'z'])
        free_throw_line_df2['line_id'] = 'free_throw_line2'
        free_throw_line_df2['line_group_id'] = 'free_throw_line2'

        self.court_lines_coordinates_df = pd.concat([
            court_df, three_point_line_df, backboard_df, hoop_df, free_throw_line_df, netdf,extradf,    
            three_point_line_df2, 
            backboard_df2, hoop_df2, 
            free_throw_line_df2, 
            netdf2,
            extradf2,
            halfdf,
            circledf
        ], ignore_index=True, axis=0)



    def get_coordinates(self):
        self.calculate_court_lines_coordinates()
        return self.court_lines_coordinates_df






# season = 2022
# game_code = 1
# competition_code = "E"

# shotdata = ShotData(competition_code)
# df = shotdata.get_game_shot_data(season, game_code)
# df.to_csv('euroleague.csv')
df = pd.read_csv('euroleague.csv')
teams = df['TEAM'].unique()
team1 = teams[0]
team2 = teams[1]
df['TEAMTYPE'] = np.where(df['TEAM'] == team1, 'Home', 'Away')
df['SHOT_DISTANCE'] = np.sqrt((df['COORD_X'] - 0)**2 + (df['COORD_Y'] - 52)**2)
df['SHOT_DISTANCE'] = df['SHOT_DISTANCE']/30.48
court = CourtCoordinates('2023-24')
court_lines_df = court.get_coordinates()
fig = px.line_3d(
    data_frame=court_lines_df,
    x='x',
    y='y',
    z='z',
    line_group='line_group_id',
    color='line_group_id',
    color_discrete_map={
        'court': 'black',
        'hoop': '#e47041',
        'net': '#D3D3D3',
        'backboard': 'gray',
        'free_throw_line': 'black',
        'hoop2':'white',
        'threepoint': 'black',
        'threepoint2' : 'black',
        'backboard2' : 'gray',
        'hoop2.1' : '#e47041',
        'hoop2.2' : 'white',
        'free_throw_line2' : 'black',
        'threepoint2.1' : 'black',
        'threepoint2.2' : 'black',
        'circlecourt' : 'black',
        'hcourt' : 'black'
    }
)
fig.update_traces(hovertemplate=None, hoverinfo='skip', showlegend=False)
fig.update_traces(line=dict(width=6))
fig.update_layout(    
    margin=dict(l=20, r=20, t=20, b=20),
    scene_aspectmode="data",
    height=600,
    scene_camera=dict(
        eye=dict(x=1.3, y=0, z=0.7)
    ),
    scene=dict(
        xaxis=dict(title='', showticklabels=False, showgrid=False,showbackground=True,backgroundcolor='black'),
        yaxis=dict(title='', showticklabels=False, showgrid=False,showbackground=True,backgroundcolor='black'),
        zaxis=dict(title='',  showticklabels=False, showgrid=False, showbackground=True, backgroundcolor='#d2a679'),
    ),
    showlegend=False,
    legend=dict(
        yanchor='top',
        y=0.05,
        x=0.2,
        xanchor='left',
        orientation='h',
        font=dict(size=15, color='gray'),
        bgcolor='rgba(0, 0, 0, 0)',
        title='',
        itemsizing='constant'
    )
)
x_values = []
y_values = []
z_values = []

for index, row in df.iterrows():
    
    if row['TEAMTYPE'] == 'Away':
        x_values.append(-row['COORD_X'])
        # Append the value from column 'x' to the list
        y_values.append(2833-row['COORD_Y']-100)
        z_values.append(0)
    else:
        x_values.append(row['COORD_X'])
        # Append the value from column 'x' to the list
        y_values.append(row['COORD_Y']+100)
        z_values.append(0)



x_values2 = []
y_values2 = []
z_values2 = []
for index, row in df.iterrows():
    # Append the value from column 'x' to the list
    

    x_values2.append(court.hoop_loc_x)
    if row['TEAMTYPE'] == 'Away':
        y_values2.append(2833-court.hoop_loc_y)
    else:
        y_values2.append(court.hoop_loc_y)
    z_values2.append(100*2)

import numpy as np
import plotly.graph_objects as go
import math
def calculate_distance(x1, y1, x2, y2):
    """Calculate the distance between two points (x1, y1) and (x2, y2)."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def generate_arc_points(p1, p2, apex, num_points=100):
    """Generate points on a quadratic Bezier curve (arc) between p1 and p2 with an apex."""
    t = np.linspace(0, 1, num_points)
    x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
    y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
    z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
    return x, y, z

# Example lists of x and y coordinates
x_coords = x_values
y_coords = y_values
z_value = 0  # Fixed z value
x_coords2 = x_values2
y_coords2 = y_values2
z_value2 = 305
for i in range(len(df)):
    x1 = x_coords[i]
    y1 = y_coords[i]
    x2 = x_coords2[i]
    y2 = y_coords2[i]
    # Define the start and end points
    p2 = np.array([x1, y1, z_value])
    p1 = np.array([x2, y2, z_value2])
    
    # Apex will be above the line connecting p1 and p2
    distance = calculate_distance(x1, y1, x2, y2)
    if df['TEAMTYPE'].iloc[i] == 'Home':
        color = 'blue'
    else:
        color = 'red'
    if df['SHOT_MADE_FLAG'].iloc[i] == 1:
        s = 'circle-open'
        s2 = 'circle'
        size = 9*1.25
        # color = 'green'
    else:
        s = 'cross'
        s2 = 'cross'
        size = 10*1.25
        # color = 'red'
    hovertemplate = f'{round(df['SHOT_DISTANCE'].iloc[i])} ft {df['ACTION'].iloc[i]} - {df['PLAYER'].iloc[i]} - {df['CONSOLE'].iloc[i]}'
    # hovertemplate= f"Game: {df['HTM'].iloc[i]} vs {df['VTM'].iloc[i]}<br>Result: {df['EVENT_TYPE'].iloc[i]}<br>Shot Type: {df['ACTION_TYPE'].iloc[i]}<br>Distance: {df['SHOT_DISTANCE'].iloc[i]} ft {df['SHOT_TYPE'].iloc[i]}<br>Quarter: {df['PERIOD'].iloc[i]}<br>Time: {df['MINUTES_REMAINING'].iloc[i]}:{df['SECONDS_REMAINING'].iloc[i]}"

    if df['SHOT_DISTANCE'].iloc[i] > 3:
        if df['SHOT_DISTANCE'].iloc[i] > 50:
            h = randint(250*3,300*3)
        elif df['SHOT_DISTANCE'].iloc[i] > 30:
            h = randint(225*3,275*3)
        elif df['SHOT_DISTANCE'].iloc[i] > 25:
            h = randint(200*3,250*3)
        elif df['SHOT_DISTANCE'].iloc[i] > 15:
            h = randint(175*3,200*3)
        else:
            h = randint(150*3,175*3)
    
        if df['SHOT_MADE_FLAG'].iloc[i] == 1:
            # h = randint(200*3,250*3)

            apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])  # Adjust apex height as needed
            
            # Generate arc points
            x, y, z = generate_arc_points(p1, p2, apex)
            fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='lines',
                        line=dict(width=8*1.25,color = color),
                        opacity =0.5,
                        name=f'Arc {i + 1}',
                        hoverinfo='text',
                        hovertemplate=hovertemplate
                    ))
# Add start and end points

    fig.add_trace(go.Scatter3d(
        x=[p2[0], p2[0]],
        y=[p2[1], p2[1]],
        z=[p2[2], p2[2]],
        mode='markers',
        marker=dict(size=size, symbol=s,color=color),
        name=f'Endpoints {i + 1}',
        hoverinfo='text',
        hovertemplate=hovertemplate
    ))
    fig.add_trace(go.Scatter3d(
        x=[p2[0], p2[0]],
        y=[p2[1], p2[1]],
        z=[p2[2], p2[2]],
        mode='markers',
        marker=dict(size=5*1.25, symbol=s2,color = color),
        name=f'Endpoints {i + 1}',
        hoverinfo='text',
        hovertemplate=hovertemplate

    ))
st.plotly_chart(fig)
# import pandas as pd

# # Load the DataFrame
# df = pd.read_csv('euroleague.csv')

# # # Initialize the SHOT_MADE_FLAG column with default values
# # df['SHOT_MADE_FLAG'] = 1  # Assuming made shots by default

# # # Iterate through the rows
# # for index, row in df.iterrows():
# #     if 'Missed' in row['ACTION']:
# #         df.at[index, 'SHOT_MADE_FLAG'] = 0  # Set to 0 if the shot was missed
# df = df[~df['ACTION'].str.contains('Free Throw', na=False)]
# df.to_csv('euroleague.csv')

