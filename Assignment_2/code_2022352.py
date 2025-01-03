# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df
    for _, it in df_trips.iterrows():
        tid = it['trip_id']
        rid = it['route_id']
        trip_to_route[tid] = rid
    for _, it in df_stop_times.iterrows():
        tid = it['trip_id']
        sid = it['stop_id']
        rid = trip_to_route.get(tid)
        if sid not in route_to_stops[rid]:
            route_to_stops[rid].append(sid)
    for sid in df_stop_times['stop_id']:
        stop_trip_count[sid] += 1
    for _, it in df_fare_rules.iterrows():
        rid = it['route_id']
        fid = it['fare_id']
        fare_rules[rid] = fid
    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id', how='left')
    # Create trip_id to route_id mapping
    # Map route_id to a list of stops in order of their sequence
    # Ensure each route only has unique stops
    # Count trips per stop
    # Create fare rules for routes
    # Merge fare rules and attributes into a single DataFrame

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    rc = defaultdict(int)
    for route_id in trip_to_route.values():
        rc[route_id] += 1  
    result = sorted(rc.items(), key=lambda x: x[1], reverse=True)[:5]
    ans = [(rid, trip_count) for rid, trip_count in result]
    return ans

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    ans = sorted(stop_trip_count.items(), key=lambda x: x[1], reverse=True)[:5]
    return ans
# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    sr = defaultdict(set)
    for rid, stops in route_to_stops.items():
        for sid in stops:
            sr[sid].add(rid)  
    res = {sid: len(routes) for sid, routes in sr.items()}
    sorted_res = sorted(res.items(), key=lambda x: x[1], reverse=True)[:5]
    return sorted_res
# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    # Dictionary to track pairs of stops and the route connecting them
    sp = defaultdict(set)
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            x = stops[i]
            y = stops[i + 1]
            sp[(x,y)].add(route_id)
    ur= []
    for stop_pair, routes in sp.items():
        if len(routes) == 1:  
            route_id = next(iter(routes))
            combined_trip_count = stop_trip_count[stop_pair[0]] + stop_trip_count[stop_pair[1]]
            ur.append((stop_pair, route_id, combined_trip_count))
    ans = sorted(ur, key=lambda x: x[2], reverse=True)[:5]
    return [(pair, route_id) for pair, route_id, _ in ans]


# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
# def visualize_stop_route_graph_interactive(route_to_stops):
def visualize_stop_route_graph_interactive():
    """
    Visualize the stop-route graph using Plotly and save it as a PNG using Matplotlib.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    sc = df_stops.set_index('stop_id')[['stop_lat', 'stop_lon', 'stop_name']].to_dict('index')
    G = nx.DiGraph()
    for stops in route_to_stops.values():
        for i in range(len(stops) - 1):
            G.add_edge(stops[i], stops[i + 1])
    ex = []
    ey = []
    for edge in G.edges():
        x0, y0 = sc[edge[0]]['stop_lon'], sc[edge[0]]['stop_lat']
        x1, y1 = sc[edge[1]]['stop_lon'], sc[edge[1]]['stop_lat']
        ex.extend([x0, x1, None])
        ey.extend([y0, y1, None])
    et = go.Scatter(x=ex, y=ey,line=dict(width=0.8, color='#880'),hoverinfo='none',mode='lines')
    n_x = []
    n_y = []
    nt = []
    for node in G.nodes():
        x, y = sc[node]['stop_lon'], sc[node]['stop_lat']
        x.append(x)
        n_y.append(y)
        nt.append(f"Stop Name: {sc[node]['stop_name']}") 
    node_trace = go.Scatter(x=n_x, y=n_y,mode='markers',text=nt,hoverinfo='text',marker=dict(color='black',size=7,line_width=1))
    fig = go.Figure(data=[et, nt],layout=go.Layout(title='graph',titlefont_size=16,showlegend=False,hovermode='closest',margin=dict(b=0, l=0, r=0, t=40),xaxis=dict(title='Longitude', showgrid=False, zeroline=False),yaxis=dict(title='Latitude', showgrid=False, zeroline=False)))
    fig.show()
    fig.write_html("stop_route_graph_2d.html")

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    ans = [] 
    for route_id, stops in route_to_stops.items():
        if (start_stop in stops) and (end_stop in stops):
                ans.append(route_id) 
    return ans

# Initialize Datalog predicates for reasoning

pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')
pyDatalog.create_terms('BoardRoute, TransferRoute')  

def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print
    DirectRoute(X, Y,R) <= RouteHasStop(R, X) & RouteHasStop(R, Y)
    OptimalRoute(X, Y, R1, Z, R2) <= (DirectRoute(X, Z, R1) & DirectRoute(Z, Y, R2)& (R1 != R2))
    BoardRoute(R, X) <= RouteHasStop(R, X)
    TransferRoute(R1, R2, Z) <= RouteHasStop(R1, Z) & RouteHasStop(R2, Z) & (R1 != R2)
    create_kb() 
    add_route_data(route_to_stops) 
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            +RouteHasStop(route_id, stop_id)


# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.
    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.
    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """  
    Ans = DirectRoute(start, end, R).data
    ans2 = [it[0] for it in Ans]
    return ans2
# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    Ans = OptimalRoute(start_stop_id, end_stop_id, R1, stop_id_to_include, R2).data
    ans2=[(route[0], stop_id_to_include, route[1]) for route in Ans]
    return ans2  

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    Ans = OptimalRoute(end_stop_id, start_stop_id, R1, stop_id_to_include, R2).data
    ans2 = [(route[0], stop_id_to_include, route[1]) for route in Ans]
    return ans2

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    Ans = (BoardRoute(R1, start_stop_id) & TransferRoute(R1, R2, stop_id_to_include) & BoardRoute(R2, end_stop_id)).data
    ans2 = [(route[0], stop_id_to_include, route[1]) for route in Ans]
    return ans2
# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    p_df = merged_fare_df[merged_fare_df['price'] <= initial_fare]
    return p_df
# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    route_summary = {}
    for rid in pruned_df['route_id'].unique():
        route_data = pruned_df[pruned_df['route_id'] == rid]
        min_price = route_data['price'].min()
        stops = set(route_to_stops.get(rid, []))
        route_summary[rid] = { 'min_price': min_price, 'stops': stops }   
    return route_summary
# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    frontier = deque([(start_stop_id, [], 0, 0)])  
    vis = set([start_stop_id])
    while frontier:
        curr, path, t ,pc  = frontier.popleft()
        if curr == end_stop_id:
            return path
        if t > max_transfers:
            continue
        for route_id, summary in route_summary.items():
            if curr in summary['stops']:
                for stop in summary['stops']:
                    if pc + summary['min_price'] <= initial_fare and stop not in vis: 
                        vis.add(stop)
                        frontier.append((stop, path + [(route_id, stop)], t+ 1, pc + summary['min_price']))
    return [] 
