import requests
import pandas as pd
import json
import io
import json
import numpy as np
import pandas as pd
import datetime
import requests
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime
from datetime import time
import time
from random import randint
import os
import numpy as np
import pandas as pd
import os
import numpy as np
import ast
import streamlit as st

st.set_page_config(page_title="3D Basketball Shot Visualizer", page_icon='🏀',layout="wide")
st.write('Enter a game id to view the 3D shot chart. Leagues such as the NBL, BAL, TBL along with some FIBA games are available to view.')
id = st.number_input('',value=1816801)
gameid = id
url = f"https://fibalivestats.dcd.shared.geniussports.com/data/{gameid}/data.json"

try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
    
    data2 = response.json()  # Parse the JSON response
    
    
    # Convert the relevant part of the JSON to a DataFrame
    # Modify the key 'your_key_here' according to the structure of your JSON
    # df = pd.DataFrame(data['playbyplay'])  # Replace 'your_key_here' with the actual key
    
    # print("Keys in the JSON data:")
    # print(data.keys())
    # df = pd.DataFrame(data['tm']['2'])
    # print(data['tm'].keys())
    # print(data2['tm']['2'])
    # data = data2['tm']['1']
    # data2.to_csv('fiba.csv')
    data_team_1 = data2['tm']['1']['shot']
    data_team_2 = data2['tm']['2']['shot']

    # Creating DataFrames for each team
    df_team_1 = pd.DataFrame(data_team_1)
    df_team_2 = pd.DataFrame(data_team_2)

    teamdf = data2['tm']['1']
    teamdf2 = data2['tm']['2']
    filtered_data = {
    'name': teamdf['name'],
    'nameInternational': teamdf['nameInternational'],
    'logo': teamdf['logo'],
    'logoT': teamdf['logoT'],
    'code': teamdf['code'],
}
    filtered_data2 = {
    'name': teamdf2['name'],
    'nameInternational': teamdf2['nameInternational'],
    'logo': teamdf2['logo'],
    'logoT': teamdf2['logoT'],
    'code': teamdf2['code'],
}

# Create the DataFrame
    tdf = pd.DataFrame([filtered_data])  # Wrap in a list to create a DataFrame
    tdf2 = pd.DataFrame([filtered_data2])
    teamdffinal = pd.concat([tdf, tdf2], ignore_index=True)

    # Optionally, you can add a column to identify the team
    df_team_1['team'] = teamdffinal['name'].iloc[0]
    df_team_2['team'] = teamdffinal['name'].iloc[1]

    # Combine both DataFrames
    combined_df = pd.concat([df_team_1, df_team_2], ignore_index=True)

    filtered_data = {
    'score': teamdf['score'],
    'full_score': teamdf['full_score'],
    'tot_sMinutes': teamdf['tot_sMinutes'],
    'tot_sFieldGoalsMade': teamdf['tot_sFieldGoalsMade'],
    'tot_sFieldGoalsAttempted': teamdf['tot_sFieldGoalsAttempted'],
    'tot_sFieldGoalsPercentage': teamdf['tot_sFieldGoalsPercentage'],
    'tot_sThreePointersMade': teamdf['tot_sThreePointersMade'],
    'tot_sThreePointersAttempted': teamdf['tot_sThreePointersAttempted'],
    'tot_sThreePointersPercentage': teamdf['tot_sThreePointersPercentage'],
    'tot_sTwoPointersMade': teamdf['tot_sTwoPointersMade'],
    'tot_sTwoPointersAttempted': teamdf['tot_sTwoPointersAttempted'],
    'tot_sTwoPointersPercentage': teamdf['tot_sTwoPointersPercentage'],
    'tot_sFreeThrowsMade': teamdf['tot_sFreeThrowsMade'],
    'tot_sFreeThrowsAttempted': teamdf['tot_sFreeThrowsAttempted'],
    'tot_sFreeThrowsPercentage': teamdf['tot_sFreeThrowsPercentage'],
    'tot_sReboundsDefensive': teamdf['tot_sReboundsDefensive'],
    'tot_sReboundsOffensive': teamdf['tot_sReboundsOffensive'],
    'tot_sReboundsTotal': teamdf['tot_sReboundsTotal'],
    'tot_sAssists': teamdf['tot_sAssists'],
    'tot_sTurnovers': teamdf['tot_sTurnovers'],
    'tot_sSteals': teamdf['tot_sSteals'],
    'tot_sBlocks': teamdf['tot_sBlocks'],
    'tot_sBlocksReceived': teamdf['tot_sBlocksReceived'],
    'tot_sFoulsPersonal': teamdf['tot_sFoulsPersonal'],
    'tot_sFoulsOn': teamdf['tot_sFoulsOn'],
    'tot_sFoulsTotal': teamdf['tot_sFoulsTotal'],
    'tot_sPoints': teamdf['tot_sPoints'],
    'tot_sPointsFromTurnovers': teamdf['tot_sPointsFromTurnovers'],
    'tot_sPointsSecondChance': teamdf['tot_sPointsSecondChance'],
    'tot_sPointsFastBreak': teamdf['tot_sPointsFastBreak'],
    'tot_sBenchPoints': teamdf['tot_sBenchPoints'],
    'tot_sPointsInThePaint': teamdf['tot_sPointsInThePaint'],
    'tot_sTimeLeading': teamdf['tot_sTimeLeading'],
    'tot_sBiggestLead': teamdf['tot_sBiggestLead'],
    'tot_sBiggestScoringRun': teamdf['tot_sBiggestScoringRun'],
    'tot_sLeadChanges': teamdf['tot_sLeadChanges'],
    'tot_sTimesScoresLevel': teamdf['tot_sTimesScoresLevel'],
    'tot_sFoulsTeam': teamdf['tot_sFoulsTeam'],
    'tot_sReboundsTeam': teamdf['tot_sReboundsTeam'],
    'tot_sReboundsTeamDefensive': teamdf['tot_sReboundsTeamDefensive'],
    'tot_sReboundsTeamOffensive': teamdf['tot_sReboundsTeamOffensive'],
    'tot_sTurnoversTeam': teamdf['tot_sTurnoversTeam'],
}
    filtered_data2 = {
    'score': teamdf2['score'],
    'full_score': teamdf2['full_score'],
    'tot_sMinutes': teamdf2['tot_sMinutes'],
    'tot_sFieldGoalsMade': teamdf2['tot_sFieldGoalsMade'],
    'tot_sFieldGoalsAttempted': teamdf2['tot_sFieldGoalsAttempted'],
    'tot_sFieldGoalsPercentage': teamdf2['tot_sFieldGoalsPercentage'],
    'tot_sThreePointersMade': teamdf2['tot_sThreePointersMade'],
    'tot_sThreePointersAttempted': teamdf2['tot_sThreePointersAttempted'],
    'tot_sThreePointersPercentage': teamdf2['tot_sThreePointersPercentage'],
    'tot_sTwoPointersMade': teamdf2['tot_sTwoPointersMade'],
    'tot_sTwoPointersAttempted': teamdf2['tot_sTwoPointersAttempted'],
    'tot_sTwoPointersPercentage': teamdf2['tot_sTwoPointersPercentage'],
    'tot_sFreeThrowsMade': teamdf2['tot_sFreeThrowsMade'],
    'tot_sFreeThrowsAttempted': teamdf2['tot_sFreeThrowsAttempted'],
    'tot_sFreeThrowsPercentage': teamdf2['tot_sFreeThrowsPercentage'],
    'tot_sReboundsDefensive': teamdf2['tot_sReboundsDefensive'],
    'tot_sReboundsOffensive': teamdf2['tot_sReboundsOffensive'],
    'tot_sReboundsTotal': teamdf2['tot_sReboundsTotal'],
    'tot_sAssists': teamdf2['tot_sAssists'],
    'tot_sTurnovers': teamdf2['tot_sTurnovers'],
    'tot_sSteals': teamdf2['tot_sSteals'],
    'tot_sBlocks': teamdf2['tot_sBlocks'],
    'tot_sBlocksReceived': teamdf2['tot_sBlocksReceived'],
    'tot_sFoulsPersonal': teamdf2['tot_sFoulsPersonal'],
    'tot_sFoulsOn': teamdf2['tot_sFoulsOn'],
    'tot_sFoulsTotal': teamdf2['tot_sFoulsTotal'],
    'tot_sPoints': teamdf2['tot_sPoints'],
    'tot_sPointsFromTurnovers': teamdf2['tot_sPointsFromTurnovers'],
    'tot_sPointsSecondChance': teamdf2['tot_sPointsSecondChance'],
    'tot_sPointsFastBreak': teamdf2['tot_sPointsFastBreak'],
    'tot_sBenchPoints': teamdf2['tot_sBenchPoints'],
    'tot_sPointsInThePaint': teamdf2['tot_sPointsInThePaint'],
    'tot_sTimeLeading': teamdf2['tot_sTimeLeading'],
    'tot_sBiggestLead': teamdf2['tot_sBiggestLead'],
    'tot_sBiggestScoringRun': teamdf2['tot_sBiggestScoringRun'],
    'tot_sLeadChanges': teamdf2['tot_sLeadChanges'],
    'tot_sTimesScoresLevel': teamdf2['tot_sTimesScoresLevel'],
    'tot_sFoulsTeam': teamdf2['tot_sFoulsTeam'],
    'tot_sReboundsTeam': teamdf2['tot_sReboundsTeam'],
    'tot_sReboundsTeamDefensive': teamdf2['tot_sReboundsTeamDefensive'],
    'tot_sReboundsTeamOffensive': teamdf2['tot_sReboundsTeamOffensive'],
    'tot_sTurnoversTeam': teamdf2['tot_sTurnoversTeam'],
}

# Create the DataFrame
    bdf = pd.DataFrame([filtered_data])  # Wrap in a list to create a DataFrame
    bdf2 = pd.DataFrame([filtered_data2])
    boxscoret = pd.concat([bdf, bdf2], ignore_index=True)

    playerdf = data2['tm']['1']['pl']
    playerdf2 = data2['tm']['2']
   
    pbdf = pd.DataFrame.from_dict(playerdf, orient='index')
    pbdf2 = pd.DataFrame.from_dict(playerdf2, orient='index')
    playerbox = pd.concat([pbdf, pbdf2], ignore_index=True)
   
    
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    
    # combined_df.to_csv('fiba.csv')
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

df = combined_df
teamdf = teamdffinal
st.write(teamdffinal)
st.write(combined_df)
df['y'] = 0.50 *df['y']
teams = df['team'].unique()
team1 = teams[0]
logo1 = teamdf['logoT'].iloc[0]
team2 = teams[1]
logo2 = teamdf['logoT'].iloc[1]
df['TEAMTYPE'] = np.where(df['team'] == team1, 'Home', 'Away')
# if df['x'] >= 50:
#     df['SHOT_DISTANCE'] = np.sqrt((df['x'] - 95)**2 + (df['COORD_Y'] - 25)**2)
# else:
#     df['SHOT_DISTANCE'] = np.sqrt((df['x'] - 5)**2 + (df['COORD_Y'] - 25)**2)
df['SHOT_DISTANCE'] = np.where(
    df['x'] >= 50,
    np.sqrt((df['x'] - 95)**2 + (df['y'] - 25)**2),
    np.sqrt((df['x'] - 5)**2 + (df['y'] - 25)**2)
)
# df['SHOT_DISTANCE'] = df['SHOT_DISTANCE']/30.48

import pandas as pd
import numpy as np


def create_pitch_3d():
    # Create figure
    fig = go.Figure()

    # Define scaling factor
    scale_factor = 1

    # Scaled pitch dimensions
    pitch_length = 100 * scale_factor
    pitch_width = 50 * scale_factor
    circle_radius = 8.15  # Radius of the center circle in meters

    # Plot pitch outline & centre line
    fig.add_trace(go.Scatter3d(x=[0, 0, pitch_length, pitch_length, 0, pitch_length / 2, pitch_length / 2], 
                               y=[0, pitch_width, pitch_width, 0, 0, 0, pitch_width],
                               z=[0]*7,
                               mode='lines',
                               line=dict(color='black', width=4 * scale_factor), hoverinfo='none'))
    
    theta = np.linspace(0, 2*np.pi, 100)  # Half circle
    x_circle = pitch_length / 2 + circle_radius * np.cos(theta)
    y_circle = pitch_width/2 + circle_radius * np.sin(theta)

    # Add the center circle as a scatter plot
    fig.add_trace(go.Scatter3d(
        x=x_circle,
        y=y_circle,
        z=[0] * len(x_circle),
        mode='lines',
        line=dict(color='black', width=4 * scale_factor),
        hoverinfo='none'
    ))
    x_floating_circle = 94 + 0.75 * np.cos(theta)
    y_floating_circle = 25 + 0.75 * np.sin(theta)
    z_floating_circle = [10] * len(x_floating_circle)  # z = 10 for the floating circle

    # Add the floating circle as a scatter plot
    fig.add_trace(go.Scatter3d(
        x=x_floating_circle,
        y=y_floating_circle,
        z=z_floating_circle,
        mode='lines',
        line=dict(color='orange', width=4 * scale_factor),  # Different color for distinction
        hoverinfo='none'
    ))
    x_floating_circle = 6 + 0.75 * np.cos(theta)
    y_floating_circle = 25 + 0.75 * np.sin(theta)
    z_floating_circle = [10] * len(x_floating_circle)  # z = 10 for the floating circle

    # Add the floating circle as a scatter plot
    fig.add_trace(go.Scatter3d(
        x=x_floating_circle,
        y=y_floating_circle,
        z=z_floating_circle,
        mode='lines',
        line=dict(color='orange', width=4 * scale_factor),  # Different color for distinction
        hoverinfo='none'
    ))
    
        # Set axis
    fig.update_layout(scene=dict(
        xaxis=dict(
            range=[-5, pitch_length+5],
            title='',
            showgrid=False,        # Turn off grid
            showline=False,        # Turn off axis line
            showticklabels=False,  # Turn off tick labels
            zeroline=False,        # Turn off the zero line
        ),
        yaxis=dict(
            range=[-5, pitch_width+5],
            title='',
            showgrid=False,        # Turn off grid
            showline=False,        # Turn off axis line
            showticklabels=False,  # Turn off tick labels
            zeroline=False,        # Turn off the zero line
        ),
        zaxis=dict(
            # range=[0, 1],
            title='',
            showgrid=False,        # Turn off grid
            showline=False,        # Turn off axis line
            showticklabels=False,  # Turn off tick labels
            zeroline=False,        # Turn off the zero line
            showbackground=True,   # Optionally keep the background
            backgroundcolor='#d2a679'
        ),
        # aspectmode='data',
        # aspectratio=dict(x=2, y=1.4, z=0.2),  # Adjust aspect ratio if needed
    ), showlegend=False,
     margin=dict(l=20, r=20, t=20, b=20),
            scene_aspectmode="data",
            height=800,
            scene_camera=dict(
                eye=dict(x=0, y=3, z=0.7)
            ),)
    
    y_lines = [50-3.5, 3.5]
    for y_line in y_lines:
        fig.add_trace(go.Scatter3d(
            x=[0, 10.5],  # Adjust according to your x range
            y=[y_line, y_line],
            z=[0, 0],  # Adjust according to your z range
            mode='lines',
            line=dict(color='black', width=4),  # Line color and width
            name=f'Line at y={y_line}'  # Optional legend entry
        ))

    y_lines = [50-3.5, 3.5]
    for y_line in y_lines:
        fig.add_trace(go.Scatter3d(
            x=[100, 100-10.5],  # Adjust according to your x range
            y=[y_line, y_line],
            z=[0, 0],  # Adjust according to your z range
            mode='lines',
            line=dict(color='black', width=4),  # Line color and width
            name=f'Line at y={y_line}'  # Optional legend entry
        ))

    x_start, y_start = 10.5, 46.5
    x_end, y_end = 10.5, 3.5
    z_value = 0  # Z-value for the half-circle

    # Calculate the radius and center for the half-circle
    radius = np.linalg.norm([x_end - x_start, y_end - y_start]) / 2
    center_x = (x_start + x_end) / 2
    center_y = (y_start + y_end) / 2

    # Rotation angle in radians
    rotation_angle = np.radians(-90)  # Rotate by -90 degrees

    # Generate points for the half-circle
    theta = np.linspace(0, np.pi, 100)  # Half-circle from 0 to π
    flattening_factor = 0.80  # Adjust this factor to make the circle flatter
    x_circle = center_x + radius * np.cos(theta)  # X values
    y_circle = center_y + (radius * flattening_factor) * np.sin(theta)  # Flatter Y values
    z_circle = np.full_like(x_circle, z_value)  # Constant z-value of 0

    # Apply rotation
    x_rotated = (x_circle - center_x) * np.cos(rotation_angle) - (y_circle - center_y) * np.sin(rotation_angle) + center_x
    y_rotated = (x_circle - center_x) * np.sin(rotation_angle) + (y_circle - center_y) * np.cos(rotation_angle) + center_y

    # Add the rotated half-circle to the plot
    fig.add_trace(go.Scatter3d(
        x=x_rotated,
        y=y_rotated,
        z=z_circle,
        mode='lines',
        line=dict(color='black', width=4),
        hoverinfo='none'
    ))

    x_start, y_start = 89.5, 46.5
    x_end, y_end = 89.5, 3.5
    z_value = 0  # Z-value for the half-circle

    # Calculate the radius and center for the half-circle
    radius = np.linalg.norm([x_end - x_start, y_end - y_start]) / 2
    center_x = (x_start + x_end) / 2
    center_y = (y_start + y_end) / 2

    # Rotation angle in radians
    rotation_angle = np.radians(90)  # Rotate by -90 degrees

    # Generate points for the half-circle
    theta = np.linspace(0, np.pi, 100)  # Half-circle from 0 to π
    flattening_factor = 0.80  # Adjust this factor to make the circle flatter
    x_circle = center_x + radius * np.cos(theta)  # X values
    y_circle = center_y + (radius * flattening_factor) * np.sin(theta)  # Flatter Y values
    z_circle = np.full_like(x_circle, z_value)  # Constant z-value of 0

    # Apply rotation
    x_rotated = (x_circle - center_x) * np.cos(rotation_angle) - (y_circle - center_y) * np.sin(rotation_angle) + center_x
    y_rotated = (x_circle - center_x) * np.sin(rotation_angle) + (y_circle - center_y) * np.cos(rotation_angle) + center_y

    # Add the rotated half-circle to the plot
    fig.add_trace(go.Scatter3d(
        x=x_rotated,
        y=y_rotated,
        z=z_circle,
        mode='lines',
        line=dict(color='black', width=4),
        hoverinfo='none'
    ))
    fig.update_layout(
        title='',
        scene=dict(
            xaxis=dict(
                range=[0, 100],
                title='',
                tickvals=list(range(0, 93, 1)),  # Tick marks at intervals of 1
                ticktext=[str(i) for i in range(0, 93, 1)]  # Corresponding labels
            ),
            yaxis=dict(
                range=[0, 50],
                title='',
                tickvals=list(range(0, 50, 1)),  # Tick marks at intervals of 1
                ticktext=[str(i) for i in range(0, 50, 1)]  # Corresponding labels
            ),
            zaxis=dict(
                # range=[0, 10],
                title='',
                tickvals=list(range(0, 11, 1)),  # Tick marks at intervals of 1
                ticktext=[str(i) for i in range(0, 11, 1)]  # Corresponding labels
            )
        )
    )
    # Define corners for the rectangle at (95, 25)
    x_rect = [95, 95, 95, 95, 95]
    y_rect = [25+2.5, 25-2.5, 25-2.5, 25+2.5, 25+2.5]  # Keep y constant
    z_rect = [13.5, 13.5, 10, 10, 13.5]  # Define z levels for rectangle

    # Add rectangle lines
    fig.add_trace(go.Scatter3d(
        x=x_rect,
        y=y_rect,
        z=z_rect,
        mode='lines',
        line=dict(color='gray', width=4),
        hoverinfo='none'
    ))
    x_rect2 = [5, 5, 5, 5, 5]

    # Add rectangle lines
    fig.add_trace(go.Scatter3d(
        x=x_rect2,
        y=y_rect,
        z=z_rect,
        mode='lines',
        line=dict(color='gray', width=4),
        hoverinfo='none'
    ))
    line_x = [100, 79.5]
    line_x2 = [0,20.5]
    line_y = [17, 17]
    line_y2 = [33,33]
    line_z = [0, 0]  # You can set z to any level you prefer

    fig.add_trace(go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode='lines',
        line=dict(color='black', width=4),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=line_x,
        y=line_y2,
        z=line_z,
        mode='lines',
        line=dict(color='black', width=4),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=[20.5,20.5],
        y=[33,17],
        z=line_z,
        mode='lines',
        line=dict(color='black', width=4),
        hoverinfo='none'
))
    

    fig.add_trace(go.Scatter3d(
        x=line_x2,
        y=line_y,
        z=line_z,
        mode='lines',
        line=dict(color='black', width=4),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=line_x2,
        y=line_y2,
        z=line_z,
        mode='lines',
        line=dict(color='black', width=4),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter3d(
        x=[79.5,79.5],
        y=[33,17],
        z=line_z,
        mode='lines',
        line=dict(color='black', width=4),
        hoverinfo='none'
))
    
    x_start, y_start = 20.5, 31
    x_end, y_end = 20.5, 19
    z_value = 0  # Z-value for the half-circle

    # Calculate the radius and center for the half-circle
    radius = np.linalg.norm([x_end - x_start, y_end - y_start]) / 2
    center_x = (x_start + x_end) / 2
    center_y = (y_start + y_end) / 2

    # Rotation angle in radians
    rotation_angle = np.radians(-90)  # Rotate by -90 degrees

    # Generate points for the half-circle
    theta = np.linspace(0, np.pi, 100)  # Half-circle from 0 to π
    flattening_factor = 0.80  # Adjust this factor to make the circle flatter
    x_circle = center_x + radius * np.cos(theta)  # X values
    y_circle = center_y + (radius * flattening_factor) * np.sin(theta)  # Flatter Y values
    z_circle = np.full_like(x_circle, z_value)  # Constant z-value of 0

    # Apply rotation
    x_rotated = (x_circle - center_x) * np.cos(rotation_angle) - (y_circle - center_y) * np.sin(rotation_angle) + center_x
    y_rotated = (x_circle - center_x) * np.sin(rotation_angle) + (y_circle - center_y) * np.cos(rotation_angle) + center_y

    # Add the rotated half-circle to the plot
    fig.add_trace(go.Scatter3d(
        x=x_rotated,
        y=y_rotated,
        z=z_circle,
        mode='lines',
        line=dict(color='black', width=4),
       hoverinfo='none'
    ))
    x_start2, y_start2 = 79.5, 31
    x_end2, y_end2 = 79.5, 19
    z_value2 = 0  # Z-value for the half-circle

    # Calculate the radius and center for the half-circle
    radius = np.linalg.norm([x_end2 - x_start2, y_end2 - y_start2]) / 2
    center_x = (x_start2 + x_end2) / 2
    center_y = (y_start2 + y_end2) / 2

    # Rotation angle in radians
    rotation_angle = np.radians(-90)  # Rotate by -90 degrees

    # Generate points for the half-circle
    theta = np.linspace(np.pi, 2*np.pi, 100)  # Half-circle from 0 to π
    flattening_factor = 0.80  # Adjust this factor to make the circle flatter
    x_circle = center_x + radius * np.cos(theta)  # X values
    y_circle = center_y + (radius * flattening_factor) * np.sin(theta)  # Flatter Y values
    z_circle = np.full_like(x_circle, z_value)  # Constant z-value of 0

    # Apply rotation
    x_rotated = (x_circle - center_x) * np.cos(rotation_angle) - (y_circle - center_y) * np.sin(rotation_angle) + center_x
    y_rotated = (x_circle - center_x) * np.sin(rotation_angle) + (y_circle - center_y) * np.cos(rotation_angle) + center_y

    # Add the rotated half-circle to the plot
    fig.add_trace(go.Scatter3d(
        x=x_rotated,
        y=y_rotated,
        z=z_circle,
        mode='lines',
        line=dict(color='black', width=4),
        hoverinfo='none'
    ))


    x_start, y_start = 6, 29
    x_end, y_end = 6, 21
    z_value = 0  # Z-value for the half-circle

    # Calculate the radius and center for the half-circle
    radius = np.linalg.norm([x_end - x_start, y_end - y_start]) / 2
    center_x = (x_start + x_end) / 2
    center_y = (y_start + y_end) / 2

    # Rotation angle in radians
    rotation_angle = np.radians(-90)  # Rotate by -90 degrees

    # Generate points for the half-circle
    theta = np.linspace(0, np.pi, 100)  # Half-circle from 0 to π
    flattening_factor = 0.95  # Adjust this factor to make the circle flatter
    x_circle = center_x + radius * np.cos(theta)  # X values
    y_circle = center_y + (radius * flattening_factor) * np.sin(theta)  # Flatter Y values
    z_circle = np.full_like(x_circle, z_value)  # Constant z-value of 0

    # Apply rotation
    x_rotated = (x_circle - center_x) * np.cos(rotation_angle) - (y_circle - center_y) * np.sin(rotation_angle) + center_x
    y_rotated = (x_circle - center_x) * np.sin(rotation_angle) + (y_circle - center_y) * np.cos(rotation_angle) + center_y

    # Add the rotated half-circle to the plot
    fig.add_trace(go.Scatter3d(
        x=x_rotated,
        y=y_rotated,
        z=z_circle,
        mode='lines',
        line=dict(color='black', width=4),
        hoverinfo='none'
    ))
    x_start2, y_start2 = 94, 29
    x_end2, y_end2 = 94, 21
    z_value2 = 0  # Z-value for the half-circle

    # Calculate the radius and center for the half-circle
    radius = np.linalg.norm([x_end2 - x_start2, y_end2 - y_start2]) / 2
    center_x = (x_start2 + x_end2) / 2
    center_y = (y_start2 + y_end2) / 2

    # Rotation angle in radians
    rotation_angle = np.radians(-90)  # Rotate by -90 degrees

    # Generate points for the half-circle
    theta = np.linspace(np.pi, 2*np.pi, 100)  # Half-circle from 0 to π
    flattening_factor = 0.95  # Adjust this factor to make the circle flatter
    x_circle = center_x + radius * np.cos(theta)  # X values
    y_circle = center_y + (radius * flattening_factor) * np.sin(theta)  # Flatter Y values
    z_circle = np.full_like(x_circle, z_value)  # Constant z-value of 0

    # Apply rotation
    x_rotated = (x_circle - center_x) * np.cos(rotation_angle) - (y_circle - center_y) * np.sin(rotation_angle) + center_x
    y_rotated = (x_circle - center_x) * np.sin(rotation_angle) + (y_circle - center_y) * np.cos(rotation_angle) + center_y

    # Add the rotated half-circle to the plot
    fig.add_trace(go.Scatter3d(
        x=x_rotated,
        y=y_rotated,
        z=z_circle,
        mode='lines',
        line=dict(color='black', width=4),
        hoverinfo='none'
    ))
    
    return fig





fig = create_pitch_3d()
x_coords = df['x']
y_coords = df['y']
z_coords = df['z'] if 'z' in df.columns else [0] * len(df)  # Default z if not available
type = df['actionType']
# hover_text = df.apply(lambda row: f"Action: {row['actionType']}<br>X: {row['x']}<br>Y: {row['y']} Team: {row['team']}", axis=1)

# x_coords = [0, 100,50,50]
# y_coords = [0, 100,50,75]
# z_coords = [0, 0,0,0]  # You need z values; this example uses 0 for both points

# Create a figure
r_values = df['r'].values  # Get the 'r' column values
teamvalues = df['team'].values
teams = df['team'].unique()
team1 = teams[0]
team2 = teams[1]
marker_symbols = ['x' if r == 0 else 'circle' for r in r_values]
sizes = [3 if r == 0 else 6 for r in r_values]
colors = ['blue' if t == team1 else 'red' for t in teamvalues]
hovertexts = df['actionType'].values

for x, y, z, symbol,size,color,hovertext in zip(x_coords, y_coords, z_coords, marker_symbols, sizes,colors,hovertexts):
    fig.add_trace(go.Scatter3d(
        x=[x],
        y=[y],
        z=[z],
        mode='markers',
        marker=dict(
            size=size,
            symbol=symbol,  # Set marker symbol conditionally
            color=color,    # Color of the markers
            opacity=0.8
        ),
        hoverinfo='text',
        text=hovertext
    ))


x_values = []
y_values = []
z_values = []
df = df[df['r'] == 1]
df = df[df['SHOT_DISTANCE'] >= 3]
for index, row in df.iterrows():
    
    if row['x'] == 'Away':
        x_values.append(row['x'])
        # Append the value from column 'x' to the list
        y_values.append(row['y'])
        z_values.append(0)
    else:
        x_values.append(row['x'])
        # Append the value from column 'x' to the list
        y_values.append(row['y'])
        z_values.append(0)



x_values2 = []
y_values2 = []
z_values2 = []
for index, row in df.iterrows():
    # Append the value from column 'x' to the list
    

    y_values2.append(25)
    if row['x'] >= 50:
        x_values2.append(94)
    else:
        x_values2.append(6)
    z_values2.append(10)

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
z_value2 = 10
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
    # if df['TEAMTYPE'].iloc[i] == 'Home':
    #     color = 'blue'
    # else:
    #     color = 'red'
    # if df['SHOT_MADE_FLAG'].iloc[i] == 1:
    #     s = 'circle-open'
    #     s2 = 'circle'
    #     size = 9
    #     # color = 'green'
    # else:
    #     s = 'cross'
    #     s2 = 'cross'
    #     size = 10*1.25
        # color = 'red'
    # hovertemplate = f'{round(df['SHOT_DISTANCE'].iloc[i])} ft {df['ACTION'].iloc[i]} - {df['PLAYER'].iloc[i]} - {df['CONSOLE'].iloc[i]}'
    # hovertemplate= f"Game: {df['HTM'].iloc[i]} vs {df['VTM'].iloc[i]}<br>Result: {df['EVENT_TYPE'].iloc[i]}<br>Shot Type: {df['ACTION_TYPE'].iloc[i]}<br>Distance: {df['SHOT_DISTANCE'].iloc[i]} ft {df['SHOT_TYPE'].iloc[i]}<br>Quarter: {df['PERIOD'].iloc[i]}<br>Time: {df['MINUTES_REMAINING'].iloc[i]}:{df['SECONDS_REMAINING'].iloc[i]}"

    if df['SHOT_DISTANCE'].iloc[i] > 3:
        if df['SHOT_DISTANCE'].iloc[i] > 50:
            h = randint(25,30)
        elif df['SHOT_DISTANCE'].iloc[i] > 30:
            h = randint(22,27)
        elif df['SHOT_DISTANCE'].iloc[i] > 25:
            h = randint(20,25)
        elif df['SHOT_DISTANCE'].iloc[i] > 15:
            h = randint(17,20)
        else:
            h = randint(15,17)
    
        


    apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])  # Adjust apex height as needed
    
    # Generate arc points
    x, y, z = generate_arc_points(p1, p2, apex)
    team = df['team'].iloc[i]
    if team == team2:
        color = 'red'
    else:
        color = 'blue'
    hovertext = df['actionType'].iloc[i]

    
    fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(width=8,color = color),
                opacity =0.5,
                name=f'Arc {i + 1}',
                hoverinfo='text',
                hovertemplate=hovertext
            ))

# Show the plot
col1, col2 = st.columns(2)
with col1:

    # # Extract the URL
    url = logo1['url']
    st.image(url,width=250)
with col2: 
 
    url = logo2['url']
    st.image(url,width=250)
st.subheader(f'{team1.title()} vs {team2.title()}')
st.plotly_chart(fig,use_container_width=True)
st.write(boxscoret)
num_players_per_row = 5
cols = st.columns(num_players_per_row)  # Initialize columns before the loop
for index, row in playerbox.iterrows():
    # Check for NaN values in the critical stats
    critical_stats = [
        'sMinutes', 
        'sFieldGoalsMade', 
        'sFieldGoalsAttempted', 
        'sFieldGoalsPercentage', 
        'sThreePointersMade', 
        'sThreePointersAttempted', 
        'sFreeThrowsMade', 
        'sReboundsTotal', 
        'sAssists', 
        'sPoints'
    ]
    
    if any(pd.isna(row[stat]) for stat in critical_stats):
        break  # Stop processing if any critical stat is NaN
    
    col_index = index % num_players_per_row  # Determine column index
    with st.expander('Player Stats'):
    
        with cols[col_index]:
            first_name = row.get('firstName', 'Unknown')
            family_name = row.get('familyName', 'Unknown')
            st.header(f"{first_name} {family_name}")
    
            # Check if the image URL exists or is valid
            if pd.notnull(row.get('photoT')) and row['photoT'].startswith("http"):
                st.image(row['photoT'], width=100)
            else:
                st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS-EnGw6bC6e96sl9Wx3B35YJajLfSN6fio4Q&s", width=100)  # Use a placeholder image
    
            st.write("### Stats")
            st.write(f"- Minutes: {row.get('sMinutes', 'N/A')}")
            st.write(f"- Field Goals Made: {row.get('sFieldGoalsMade', 'N/A')}")
            st.write(f"- Field Goals Attempted: {row.get('sFieldGoalsAttempted', 'N/A')}")
            st.write(f"- Field Goals Percentage: {row.get('sFieldGoalsPercentage', 'N/A')}%")
            st.write(f"- Three Pointers Made: {row.get('sThreePointersMade', 'N/A')}")
            st.write(f"- Three Pointers Attempted: {row.get('sThreePointersAttempted', 'N/A')}")
            st.write(f"- Free Throws Made: {row.get('sFreeThrowsMade', 'N/A')}")
            st.write(f"- Rebounds: {row.get('sReboundsTotal', 'N/A')}")
            st.write(f"- Assists: {row.get('sAssists', 'N/A')}")
            st.write(f"- Points: {row.get('sPoints', 'N/A')}")
            st.write("---")
    
        if col_index == num_players_per_row - 1:  # After filling one row
            cols = st.columns(num_players_per_row)  # Reset columns for the next row


# Show the plot
