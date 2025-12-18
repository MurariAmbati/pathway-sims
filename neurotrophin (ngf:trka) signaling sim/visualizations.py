"""
advanced visualization components for ngf/trka signaling
interactive plotly-based graphs and network diagrams
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

def create_network_graph(model, results):
    """
    create interactive 3d network graph of signaling pathways
    nodes represent proteins, edges represent interactions
    """
    G = nx.DiGraph()
    
    # define nodes with positions and attributes
    nodes = []
    edges = []
    
    # receptor layer
    nodes.append(("ngf", {"layer": 0, "pathway": "ligand", "size": 20}))
    nodes.append(("trka", {"layer": 0, "pathway": "receptor", "size": 25}))
    
    # adaptor layer
    nodes.append(("grb2/sos", {"layer": 1, "pathway": "adaptor", "size": 15}))
    nodes.append(("gab1", {"layer": 1, "pathway": "adaptor", "size": 15}))
    nodes.append(("shc", {"layer": 1, "pathway": "adaptor", "size": 15}))
    
    # pathway nodes
    if "ras/mapk" in model.pathways:
        nodes.extend([
            ("ras", {"layer": 2, "pathway": "ras/mapk", "size": 20}),
            ("raf", {"layer": 3, "pathway": "ras/mapk", "size": 18}),
            ("mek", {"layer": 4, "pathway": "ras/mapk", "size": 18}),
            ("erk", {"layer": 5, "pathway": "ras/mapk", "size": 20}),
        ])
        edges.extend([
            ("trka", "grb2/sos"), ("grb2/sos", "ras"),
            ("ras", "raf"), ("raf", "mek"), ("mek", "erk")
        ])
    
    if "pi3k/akt" in model.pathways:
        nodes.extend([
            ("pi3k", {"layer": 2, "pathway": "pi3k/akt", "size": 20}),
            ("pip3", {"layer": 3, "pathway": "pi3k/akt", "size": 15}),
            ("pdk1", {"layer": 4, "pathway": "pi3k/akt", "size": 18}),
            ("akt", {"layer": 5, "pathway": "pi3k/akt", "size": 20}),
            ("bad", {"layer": 6, "pathway": "pi3k/akt", "size": 15}),
        ])
        edges.extend([
            ("trka", "gab1"), ("gab1", "pi3k"),
            ("pi3k", "pip3"), ("pip3", "pdk1"), ("pdk1", "akt"), ("akt", "bad")
        ])
    
    if "plcγ" in model.pathways:
        nodes.extend([
            ("plcγ", {"layer": 2, "pathway": "plcγ", "size": 20}),
            ("ip3", {"layer": 3, "pathway": "plcγ", "size": 15}),
            ("dag", {"layer": 3, "pathway": "plcγ", "size": 15}),
            ("pkc", {"layer": 4, "pathway": "plcγ", "size": 18}),
            ("ca2+", {"layer": 4, "pathway": "plcγ", "size": 18}),
        ])
        edges.extend([
            ("trka", "plcγ"), ("plcγ", "ip3"), ("plcγ", "dag"),
            ("dag", "pkc"), ("ip3", "ca2+"), ("ca2+", "pkc")
        ])
    
    # add feedback edges if enabled
    if model.feedback:
        edges.append(("erk", "grb2/sos"))
        edges.append(("akt", "raf"))
    
    # add crosstalk edges if enabled
    if model.crosstalk:
        edges.append(("pkc", "raf"))
        edges.append(("ca2+", "ras"))
    
    # add nodes and edges to graph
    for node, attrs in nodes:
        G.add_node(node, **attrs)
    
    G.add_edges_from(edges)
    
    # calculate layout
    pos = nx.spring_layout(G, k=2, iterations=50, dim=3, seed=42)
    
    # extract coordinates
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    # create edge trace
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(0, 212, 255, 0.3)', width=2),
        hoverinfo='none',
        showlegend=False
    )
    
    # create node traces by pathway
    pathway_colors = {
        "ligand": "#ff00ff",
        "receptor": "#00ff00",
        "adaptor": "#ffff00",
        "ras/mapk": "#00d4ff",
        "pi3k/akt": "#ff6b35",
        "plcγ": "#8f00ff"
    }
    
    node_traces = []
    for pathway, color in pathway_colors.items():
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            if G.nodes[node].get('pathway') == pathway:
                x, y, z = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                node_text.append(node)
                node_size.append(G.nodes[node].get('size', 15))
        
        if node_x:
            trace = go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers+text',
                name=pathway,
                marker=dict(
                    size=node_size,
                    color=color,
                    line=dict(color='white', width=2)
                ),
                text=node_text,
                textposition="top center",
                textfont=dict(size=10, color='white'),
                hoverinfo='text'
            )
            node_traces.append(trace)
    
    # create figure
    fig = go.Figure(data=[edge_trace] + node_traces)
    
    fig.update_layout(
        title="signaling network topology",
        showlegend=True,
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        height=700,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(0,0,0,0.5)'
        )
    )
    
    return fig

def create_temporal_plot(time_points, results, pathways):
    """
    create multi-panel temporal dynamics plot
    shows activation profiles over time for each pathway
    """
    # determine number of subplots
    n_pathways = len(pathways)
    
    fig = make_subplots(
        rows=n_pathways,
        cols=1,
        subplot_titles=pathways,
        vertical_spacing=0.1
    )
    
    row = 1
    colors = ['#00d4ff', '#ff6b35', '#8f00ff', '#00ff00', '#ffff00']
    
    if "ras/mapk" in pathways:
        components = [
            ('ras_gtp', 'ras-gtp', colors[0]),
            ('raf_active', 'raf*', colors[1]),
            ('mek_active', 'mek*', colors[2]),
            ('erk_active', 'erk*', colors[3])
        ]
        
        for key, name, color in components:
            if key in results:
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=results[key],
                        name=name,
                        mode='lines',
                        line=dict(color=color, width=2.5),
                        legendgroup='mapk'
                    ),
                    row=row,
                    col=1
                )
        row += 1
    
    if "pi3k/akt" in pathways:
        components = [
            ('pi3k_active', 'pi3k*', colors[0]),
            ('pip3', 'pip3', colors[1]),
            ('akt_active', 'akt*', colors[2]),
            ('bad_phospho', 'bad-p', colors[3])
        ]
        
        for key, name, color in components:
            if key in results:
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=results[key],
                        name=name,
                        mode='lines',
                        line=dict(color=color, width=2.5),
                        legendgroup='akt'
                    ),
                    row=row,
                    col=1
                )
        row += 1
    
    if "plcγ" in pathways:
        components = [
            ('plcg_active', 'plcγ*', colors[0]),
            ('ip3', 'ip3', colors[1]),
            ('dag', 'dag', colors[2]),
            ('pkc_active', 'pkc*', colors[3]),
            ('calcium', 'ca2+', colors[4])
        ]
        
        for key, name, color in components:
            if key in results:
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=results[key],
                        name=name,
                        mode='lines',
                        line=dict(color=color, width=2.5),
                        legendgroup='plcg'
                    ),
                    row=row,
                    col=1
                )
        row += 1
    
    fig.update_xaxes(title_text="time (minutes)", row=n_pathways, col=1)
    fig.update_yaxes(title_text="concentration")
    
    fig.update_layout(
        height=300 * n_pathways,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_heatmap(time_points, results):
    """
    create activation heatmap showing all components over time
    """
    # select key components
    components = [
        'trka_dimer', 'ras_gtp', 'raf_active', 'mek_active', 'erk_active',
        'pi3k_active', 'pip3', 'akt_active', 'bad_phospho',
        'plcg_active', 'ip3', 'dag', 'pkc_active', 'calcium'
    ]
    
    # build matrix
    matrix = []
    labels = []
    
    for comp in components:
        if comp in results:
            # normalize to 0-100
            data = results[comp]
            normalized = 100 * (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
            matrix.append(normalized)
            labels.append(comp.replace('_', ' '))
    
    if not matrix:
        return go.Figure()
    
    matrix = np.array(matrix)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=time_points,
        y=labels,
        colorscale='Viridis',
        colorbar=dict(title='activation (%)'),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="pathway activation heatmap",
        xaxis_title="time (minutes)",
        yaxis_title="component",
        template='plotly_dark',
        height=600
    )
    
    return fig

def create_3d_surface(results):
    """
    create 3d surface plot of pathway interactions
    """
    # create synthetic landscape based on erk and akt activation
    if 'erk_active' in results and 'akt_active' in results:
        erk_data = results['erk_active']
        akt_data = results['akt_active']
        
        # handle empty or single-value arrays
        erk_max = np.max(erk_data) if len(erk_data) > 0 and np.max(erk_data) > 0 else 100
        akt_max = np.max(akt_data) if len(akt_data) > 0 and np.max(akt_data) > 0 else 100
        
        # create meshgrid
        xi = np.linspace(0, erk_max, 50)
        yi = np.linspace(0, akt_max, 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # create landscape (survival/differentiation potential)
        # nonlinear response surface with synergy term
        Zi = 0.6 * Xi + 0.4 * Yi + 0.2 * np.sqrt(Xi * Yi)
        
    else:
        Xi, Yi = np.meshgrid(np.linspace(0, 100, 50), np.linspace(0, 100, 50))
        Zi = 0.6 * Xi + 0.4 * Yi + 0.2 * np.sqrt(Xi * Yi)
    
    fig = go.Figure(data=[
        go.Surface(
            x=Xi,
            y=Yi,
            z=Zi,
            colorscale='Plasma',
            opacity=0.9,
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
            )
        )
    ])
    
    fig.update_layout(
        title="signaling landscape",
        scene=dict(
            xaxis_title="erk activation",
            yaxis_title="akt activation",
            zaxis_title="cellular response",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        template='plotly_dark',
        height=600
    )
    
    return fig

def create_phase_portrait(results):
    """
    create phase portrait of key pathway components
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("erk vs akt phase space", "calcium oscillations")
    )
    
    # erk vs akt phase portrait
    if 'erk_active' in results and 'akt_active' in results:
        fig.add_trace(
            go.Scatter(
                x=results['erk_active'],
                y=results['akt_active'],
                mode='lines+markers',
                marker=dict(
                    size=4,
                    color=np.arange(len(results['erk_active'])),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="time", x=0.45)
                ),
                line=dict(color='rgba(0, 212, 255, 0.3)', width=1),
                showlegend=False
            ),
            row=1,
            col=1
        )
    
    # calcium phase portrait
    if 'calcium' in results and 'pkc_active' in results:
        fig.add_trace(
            go.Scatter(
                x=results['calcium'],
                y=results['pkc_active'],
                mode='lines+markers',
                marker=dict(
                    size=4,
                    color=np.arange(len(results['calcium'])),
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="time", x=1.0)
                ),
                line=dict(color='rgba(143, 0, 255, 0.3)', width=1),
                showlegend=False
            ),
            row=1,
            col=2
        )
    
    fig.update_xaxes(title_text="erk activation", row=1, col=1)
    fig.update_yaxes(title_text="akt activation", row=1, col=1)
    fig.update_xaxes(title_text="calcium", row=1, col=2)
    fig.update_yaxes(title_text="pkc activation", row=1, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        height=500,
        showlegend=False
    )
    
    return fig
