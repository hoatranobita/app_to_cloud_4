import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
import plotly.figure_factory as ff
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from sklearn.decomposition import PCA

df = pd.read_table('https://raw.githubusercontent.com/PineBiotech/omicslogic/master/CellLines_15Genes_marked.txt',sep='\t',header=(0)) 

labels_list = df.loc[[0]]
gene_list = df.index

#Make a list of sample names
sample_names = df.loc[0] 
sample_names = list(df)[1:] 

#Make unique list of labels
labels = np.unique(labels_list) 

#delete "Group" label 
labels = np.delete(labels, np.where(labels == 'Group')) 

#Remove row "Group" at index 0 from data
data = df.drop([0]) 

#Remove id from axis
data.index = data['id']
data = data.drop(['id'], axis=1) 
data = data.transpose()

#Get list of all labels
flabels = labels_list.loc[0,:].values[1:]
scaled = StandardScaler() 
scaled.fit(data) 
scaled_data = scaled.transform(data)

template = 'ggplot2'

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LUX])
server = app.server
app.layout = html.Div([
    dbc.Tabs(
        [dbc.Tab(label="PCA", tab_id="pca"),
         dbc.Tab(label="Kmeans", tab_id="kmeans"),
         dbc.Tab(label="H-clust", tab_id="hclust"),
         dbc.Tab(label="Birch", tab_id="birch"),
         dbc.Tab(label="Compare", tab_id="compare"),
         dbc.Tab(label="Table and Code", tab_id="table")],
        id="tabs",
        active_tab="pca",
    ),
    html.Div(id="tab-content", className="p-4"),
])

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab")])

def render_tab_content(active_tab):
    if active_tab == "pca":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.RadioItems(
                        options=[
                            {"label": "2D", "value": '2D'},
                            {"label": "3D", "value": '3D'}],
                        value='2D',
                        id="radioitems-input",
                        inline=True
                    ),
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}
                dbc.Col([
                    html.H6('Num of PCA',style={'padding-top' : 10,'padding-right' : 2}),
                    dcc.Input(type="number",id="num_of_pca",value=4,min=3,style={'width':'20%','text-align':'center'})   
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center','display':'flex'}), #,style={'padding-top' : 10}            
                dbc.Col([
                    html.H6('x',style={'padding-top' : 10,'padding-right' : 2}),
                    dcc.Input(size="sm",type="number",id="num_of_x",value=0,min=0,style={'width':'40%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}), #,style={'padding-top' : 10}            
                dbc.Col([
                    html.H6('y',style={'padding-top' : 10,'padding-right' : 2}),
                    dcc.Input(size="sm",type="number",id="num_of_y",value=1,min=0,style={'width':'40%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}), #,style={'padding-top' : 10}             
                dbc.Col([
                    html.H6('z',style={'padding-top' : 10,'padding-right' : 2}),
                    dcc.Input(size="sm",type="number",id="num_of_z",value=2,min=0,style={'width':'40%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}), #,style={'padding-top' : 10}
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Template',style={'padding-top' : 10,'padding-right' : 2})
                        ],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="template_1",
                                         options=[{'label': 'ggplot2', 'value': 'ggplot2'},
                                                  {'label': 'plotly_white', 'value': 'plotly_white'}],
                                         value='ggplot2',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)
                        ],width={'size':7,"offset":0,'order':1},style={'text-align':'center'})
                    ])
                ],width={'size':3,"offset":0,'order':1},style={'text-align':'center'})
            ], className='p-2 align-items-stretch'),
            
            dbc.Row([      
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                             dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_1',color="secondary"),
                                    dcc.Download(id='download_1'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_2',color="secondary"),
                                    dcc.Download(id='download_2'),
                                    dbc.Button("csv", size="sm",className="me-1",id='btn_3',color="secondary"),
                                    dcc.Download(id='download_3')                               
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id='chart_title'),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='pie_chart',figure={}),
                                ],width={'size':12,'offset':0,'order':1}),
                            ]),                        
                        ])
                    ], className='h-100 text-left')
                ], xs=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_4',color="secondary"),
                                    dcc.Download(id='download_4'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_5',color="secondary"),
                                    dcc.Download(id='download_5')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                       
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f"PC Explained Variance Ratio",style={'text-align':'center'}),
                                    dcc.Graph(id='bar_chart',figure={}),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                        ])
                    ], className='h-100 text-left')
                ], xs=3),                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_6',color="secondary"),
                                    dcc.Download(id='download_6'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_7',color="secondary"),
                                    dcc.Download(id='download_7')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                      
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f'Principal Components Cumulative Variance Ratio Explained',style={'text-align':'center'}),
                                    dcc.Graph(id='bar_chart_2',figure={}),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                        ])
                    ], className='h-100 text-left')
                ], xs=3),
            
            ], className='p-2 align-items-stretch'),         
                
            dbc.Row([                  
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_8',color="secondary"),
                                    dcc.Download(id='download_8'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_9',color="secondary"),
                                    dcc.Download(id='download_9')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                       
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='pie_chart_2',figure={}),
                                ],width={'size':12,'offset':0,'order':1}),
                            ]),
                        ])
                    ], className='h-100 text-left')
                ], xs=12),
            ], className='p-2 align-items-stretch'),            
             dbc.Row([                  
                dbc.Col([
                    html.H6('Num of genes',style={'padding-top' : 10,'padding-right' : 2}),
                    dcc.Input(type="number",id="num_of_genes",value=15,min=1,style={'width':'30%','text-align':'center'})   
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center','display':'flex'}),                   
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Template',style={'padding-top' : 10,'padding-right' : 2})
                        ],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="overlay",
                                         options=[{'label': 'Overlay', 'value': 'Overlay'},
                                                  {'label': 'No Overlay', 'value': 'No Overlay'}],
                                         value='Overlay',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)
                        ],width={'size':7,"offset":0,'order':1},style={'text-align':'center'})
                    ])
                ],width={'size':3,"offset":0,'order':1},style={'text-align':'center'})
             ], className='p-2 align-items-stretch'),           

            dbc.Row([                  
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_41',color="secondary"),
                                    dcc.Download(id='download_41'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_42',color="secondary"),
                                    dcc.Download(id='download_42')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                       
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f'PCA Biplot: Principal Components Loadings (Genes)',style={'text-align':'center'}),
                                    dcc.Graph(id='bio_chart',figure={},clickData={'points': [{'hovertext': '184A1'}]}),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                        ])
                    ], className='h-100 text-left')
                ], xs=6),                     
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([                                      
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_43',color="secondary"),
                                    dcc.Download(id='download_43'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_44',color="secondary"),
                                    dcc.Download(id='download_44')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                            
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id='histogram_name',style={'text-align':'center'}),
                                    dcc.Graph(id='custom_data',figure={}),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                        ])
                    ], className='h-100 text-left')
                ], xs=6),            
            ], className='p-2 align-items-stretch'),
            dcc.Store(id='store-data', data=[], storage_type='memory'), 

        ])
           
    elif active_tab == "kmeans":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6('PCA',style={'padding-top' : 10,'padding-right' : 2}),
                    dcc.Input(type="number",id="num_of_pca_2",value=4,min=3,style={'width':'50%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}), #,style={'padding-top' : 10}
                dbc.Col([
                    html.H6('Clus',style={'padding-top' : 10,'padding-right' : 2}),
                    dcc.Input(type="number",id="num_of_cluster_2",value=6,min=6,style={'width':'60%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('x',style={'padding-top' : 10,'padding-right' : 2})],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="num_of_x_2",
                                         options=[],
                                         value='PC1',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)],width={'size':7,"offset":0,'order':1})
                    ])
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('y',style={'padding-top' : 10,'padding-right' : 2})
                        ],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="num_of_y_2",
                                         options=[],
                                         value='PC2',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)
                        ],width={'size':7,"offset":0,'order':1})
                    ])
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('z',style={'padding-top' : 10,'padding-right' : 2})],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="num_of_z_2",
                                         options=[],
                                         value='PC3',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)],width={'size':7,"offset":0,'order':1})
                    ])
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}
                dbc.Col([
                    html.H6('Range',style={'padding-top' : 10}),
                    dcc.Input(type="number",id="range_1",value=2,min=2,style={'width':'60%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}), #,style={'padding-top' : 10}
                dbc.Col([
                    dcc.Input(type="number",id="range_2",value=30,min=3,style={'width':'50%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Template',style={'padding-top' : 10,'padding-right' : 2})
                        ],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="template_2",
                                         options=[{'label': 'ggplot2', 'value': 'ggplot2'},
                                                  {'label': 'plotly_white', 'value': 'plotly_white'}],
                                         value='ggplot2',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)
                        ],width={'size':7,"offset":0,'order':1},style={'text-align':'center'})
                    ])
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'})
            ], className='p-2 align-items-stretch'),
            dbc.Row([      
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_10',color="secondary"),
                                    dcc.Download(id='download_10'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_11',color="secondary"),
                                    dcc.Download(id='download_11'),
                                    dbc.Button("CSV", size="sm",className="me-1",id='btn_12',color="secondary"),
                                    dcc.Download(id='download_12')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                        
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f'PCA for Kmeans Clusters'),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='pie_chart_3',figure={}),
                                ],width={'size':12,'offset':0,'order':1}),
                            ]),                        
                        ])
                    ], className='h-100 text-left')
                ], xs=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_13',color="secondary"),
                                    dcc.Download(id='download_13'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_14',color="secondary"),
                                    dcc.Download(id='download_14')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),  
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f"Kmeans Clustering compared with Known Labels",style={'text-align':'center'}),
                                    dcc.Graph(id='bar_chart_3',figure={}),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                        ])
                    ], className='h-100 text-left')
                ], xs=4),                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_15',color="secondary"),
                                    dcc.Download(id='download_15'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_16',color="secondary"),
                                    dcc.Download(id='download_16')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                       
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f'Clustering Silhoette',style={'text-align':'center'}),
                                    dbc.Spinner(children=[dcc.Graph(id='bar_chart_4',figure={})], size="sm", color="primary", type="border", fullscreen=False),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                        ])
                    ], className='h-100 text-left')
                ], xs=4),                

            ], className='p-2 align-items-stretch'),
        ])
            
    elif active_tab == "hclust":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6('PCA',style={'padding-top' : 10,'padding-right' : 2}),
                    dcc.Input(type="number",id="num_of_pca_3",value=4,min=3,style={'width':'50%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}), #,style={'padding-top' : 10}
                dbc.Col([
                    html.H6('Clus',style={'padding-top' : 10,'padding-right' : 2}),
                    dcc.Input(type="number",id="num_of_cluster_3",value=6,min=6,style={'width':'60%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('x',style={'padding-top' : 10,'padding-right' : 2})],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="num_of_x_3",
                                         options=[],
                                         value='PC1',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)],width={'size':7,"offset":0,'order':1})
                    ])
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('y',style={'padding-top' : 10,'padding-right' : 2})
                        ],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="num_of_y_3",
                                         options=[],
                                         value='PC2',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)
                        ],width={'size':7,"offset":0,'order':1})
                    ])
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('z',style={'padding-top' : 10,'padding-right' : 2})],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="num_of_z_3",
                                         options=[],
                                         value='PC3',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)],width={'size':7,"offset":0,'order':1})
                    ])
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}
                dbc.Col([
                    html.H6('Range',style={'padding-top' : 10}),
                    dcc.Input(type="number",id="range_3",value=2,min=2,style={'width':'60%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}), #,style={'padding-top' : 10}
                dbc.Col([
                    dcc.Input(type="number",id="range_4",value=30,min=3,style={'width':'50%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Template',style={'padding-top' : 10,'padding-right' : 2})
                        ],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="template_3",
                                         options=[{'label': 'ggplot2', 'value': 'ggplot2'},
                                                  {'label': 'plotly_white', 'value': 'plotly_white'}],
                                         value='ggplot2',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)
                        ],width={'size':7,"offset":0,'order':1},style={'text-align':'center'})
                    ])
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'})
            ], className='p-2 align-items-stretch'),
            dbc.Row([      
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_17',color="secondary"),
                                    dcc.Download(id='download_17'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_18',color="secondary"),
                                    dcc.Download(id='download_18'),
                                    dbc.Button("CSV", size="sm",className="me-1",id='btn_19',color="secondary"),
                                    dcc.Download(id='download_19')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                       
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f'PCA for H-clust Clusters'),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='pie_chart_4',figure={}),
                                ],width={'size':12,'offset':0,'order':1}),
                            ]),                        
                        ])
                    ], className='h-100 text-left')
                ], xs=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_20',color="secondary"),
                                    dcc.Download(id='download_20'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_21',color="secondary"),
                                    dcc.Download(id='download_21')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                       
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f"Hierarchical Clustering compared with Known Labels",style={'text-align':'center'}),
                                    dcc.Graph(id='bar_chart_5',figure={}),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                        ])
                    ], className='h-100 text-left')
                ], xs=4),                
                
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_22',color="secondary"),
                                    dcc.Download(id='download_23'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_23',color="secondary"),
                                    dcc.Download(id='download_23')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                       
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f'Clustering Silhoette',style={'text-align':'center'}),
                                    dbc.Spinner(children=[dcc.Graph(id='bar_chart_6',figure={})], size="sm", color="primary", type="border", fullscreen=False),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                        ])
                    ], className='h-100 text-left')
                ], xs=4),                

            ], className='p-2 align-items-stretch'),                  
        ])

    elif active_tab == "birch":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6('PCA',style={'padding-top' : 10,'padding-right' : 2}),
                    dcc.Input(type="number",id="num_of_pca_4",value=4,min=3,style={'width':'50%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}), #,style={'padding-top' : 10}
                dbc.Col([
                    html.H6('Clus',style={'padding-top' : 10,'padding-right' : 2}),
                    dcc.Input(type="number",id="num_of_cluster_4",value=6,min=6,style={'width':'60%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('x',style={'padding-top' : 10,'padding-right' : 2})],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="num_of_x_4",
                                         options=[],
                                         value='PC1',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)],width={'size':7,"offset":0,'order':1})
                    ])
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('y',style={'padding-top' : 10,'padding-right' : 2})
                        ],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="num_of_y_4",
                                         options=[],
                                         value='PC2',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)
                        ],width={'size':7,"offset":0,'order':1})
                    ])
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('z',style={'padding-top' : 10,'padding-right' : 2})],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="num_of_z_4",
                                         options=[],
                                         value='PC3',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)],width={'size':7,"offset":0,'order':1})
                    ])
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}
                dbc.Col([
                    html.H6('Range',style={'padding-top' : 10}),
                    dcc.Input(type="number",id="range_5",value=2,min=2,style={'width':'60%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}), #,style={'padding-top' : 10}
                dbc.Col([
                    dcc.Input(type="number",id="range_6",value=30,min=3,style={'width':'50%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center','display':'flex'}),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6('Template',style={'padding-top' : 10,'padding-right' : 2})
                        ],width={'size':5,"offset":0,'order':1}),
                        dbc.Col([
                            dcc.Dropdown(id="template_4",
                                         options=[{'label': 'ggplot2', 'value': 'ggplot2'},
                                                  {'label': 'plotly_white', 'value': 'plotly_white'}],
                                         value='ggplot2',
                                         multi=False,
                                         disabled=False,
                                         clearable=False,
                                         searchable=True)
                        ],width={'size':7,"offset":0,'order':1},style={'text-align':'center'})
                    ])
                ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'})
            ], className='p-2 align-items-stretch'),
            dbc.Row([      
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_24',color="secondary"),
                                    dcc.Download(id='download_24'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_25',color="secondary"),
                                    dcc.Download(id='download_25'),
                                    dbc.Button("CSV", size="sm",className="me-1",id='btn_26',color="secondary"),
                                    dcc.Download(id='download_26')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                         
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f'PCA for H-clust Clusters'),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='pie_chart_5',figure={}),
                                ],width={'size':12,'offset':0,'order':1}),
                            ]),                        
                        ])
                    ], className='h-100 text-left')
                ], xs=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_27',color="secondary"),
                                    dcc.Download(id='download_27'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_28',color="secondary"),
                                    dcc.Download(id='download_28')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                       
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f"Birch Clustering compared with Known Labels",style={'text-align':'center'}),
                                    dcc.Graph(id='bar_chart_7',figure={}),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                        ])
                    ], className='h-100 text-left')
                ], xs=4),                
                
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_29',color="secondary"),
                                    dcc.Download(id='download_29'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_30',color="secondary"),
                                    dcc.Download(id='download_30')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                        
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f'Clustering Silhoette',style={'text-align':'center'}),
                                    dbc.Spinner(children=[dcc.Graph(id='bar_chart_8',figure={})], size="sm", color="primary", type="border", fullscreen=False),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                        ])
                    ], className='h-100 text-left')
                ], xs=4),                

            ], className='p-2 align-items-stretch'),                  
        ])
    
    elif active_tab == "compare":
        return html.Div([
                        dbc.Row([                 
                            dbc.Col([
                                html.H6('Check List'),
                    
                                dbc.Checklist(
                                    options=[
                                        {"label": "K-means", "value": 'Kmeans_sa'},
                                        {"label": "H-clust", "value": 'Hclust_sa'},
                                        {"label": "Birch", "value": 'Birch_sa'}],
                                    value=['Kmeans_sa','Hclust_sa','Birch_sa'],
                                    id="radioitems-input-2",
                                    inline=True
                                ),
                            ],width={'size':3,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}             
            
                            dbc.Col([
                                html.H6('Range'),
                                dcc.Input(type="number",id="range_7",value=2,min=2,style={'width':'40%','text-align':'center'})
                            ],width={'size':1,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}            
                            dbc.Col([
                                html.H6('Range'),
                                dcc.Input(type="number",id="range_8",value=30,min=3,style={'width':'50%','text-align':'center'})
                            ],width={'size':1,"offset":0,'order':1},style={'text-align':'center'}),
                            dbc.Col([
                                html.H6('PCA'),
                                dcc.Input(type="number",id="num_of_pca_5",value=4,min=3,style={'width':'40%','text-align':'center'})
                            ],width={'size':1,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}            
                            dbc.Col([
                                html.H6('Cluster'),
                                dcc.Input(type="number",id="num_of_cluster_5",value=6,min=6,style={'width':'40%','text-align':'center'})
                            ],width={'size':1,"offset":0,'order':1},style={'text-align':'center'}),
                                                        dbc.Col([
                                html.H6('x'),
                                dcc.Dropdown(id="num_of_x_5",
                                             options=[],
                                             value='PC1',
                                             multi=False,
                                             disabled=False,
                                             clearable=False,
                                             searchable=True)
                            ],width={'size':1,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10} 
                            dbc.Col([
                                html.H6('y'),
                                dcc.Dropdown(id="num_of_y_5",
                                             options=[],
                                             value='PC2',
                                             multi=False,
                                             disabled=False,
                                             clearable=False,
                                             searchable=True)
                            ],width={'size':1,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}
                            dbc.Col([
                                html.H6('z'),
                                dcc.Dropdown(id="num_of_z_5",
                                             options=[],
                                             value='PC3',
                                             multi=False,
                                             disabled=False,
                                             clearable=False,
                                             searchable=True)
                            ],width={'size':1,"offset":0,'order':1},style={'text-align':'center'}),
                            dbc.Col([
                                html.H6('Template'),
                                dcc.Dropdown(id="template_5",
                                             options=[{'label': 'ggplot2', 'value': 'ggplot2'},
                                                      {'label': 'plotly_white', 'value': 'plotly_white'}],
                                             value='ggplot2',
                                             multi=False,
                                             disabled=False,
                                             clearable=False,
                                             searchable=True)
                            ],width={'size':2,"offset":0,'order':1},style={'text-align':'center'}),                          
                        ], className='p-2 align-items-stretch'),
            dbc.Row([      
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([      
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_31',color="secondary"),
                                    dcc.Download(id='download_31'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_32',color="secondary"),
                                    dcc.Download(id='download_32'),
                                    dbc.Button("CSV", size="sm",className="me-1",id='btn_33',color="secondary"),
                                    dcc.Download(id='download_33')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                      
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f'Clustering Silhoette'),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Spinner(children=[dcc.Graph(id='pie_chart_8',figure={})], size="sm", color="primary", type="border", fullscreen=False),
                                ],width={'size':12,'offset':0,'order':1}),
                            ]),                        
                        ])
                    ], className='h-100 text-left')
                ], xs=4),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_34',color="secondary"),
                                    dcc.Download(id='download_34'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_35',color="secondary"),
                                    dcc.Download(id='download_35'),
                                    dbc.Button("CSV", size="sm",className="me-1",id='btn_36',color="secondary"),
                                    dcc.Download(id='download_36')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                       
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f"Clustering Results on PCA Scatterplot",style={'text-align':'center'}),
                                    dcc.Graph(id='bar_chart_9',figure={},style={'height':230}),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                        ])
                    ], className='h-100 text-left')
                        ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'})
                    ]),
                    dbc.Row([
                        dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_37',color="secondary"),
                                    dcc.Download(id='download_37'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_38',color="secondary"),
                                    dcc.Download(id='download_38')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                      
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f"Heatmap",style={'text-align':'center'}),
                                    dcc.Graph(id='bar_chart_10',figure={},style={'height':230}),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                        ])
                    ], className='h-100 text-left')                
                
                        ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'})
                    ])
                ], xs=8),                                             
            ], className='p-2 align-items-stretch'),
            dbc.Row([      
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([                                    
                                    dbc.Button("SVG", size="sm",className="me-1",id='btn_39',color="secondary"),
                                    dcc.Download(id='download_39'),
                                    dbc.Button("HTML", size="sm",className="me-1",id='btn_40',color="secondary"),
                                    dcc.Download(id='download_40')
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'right'}),
                            ]),                                        
                            dbc.Row([
                                dbc.Col([
                                    html.Span(f'Hclust Dendrogram'),
                                ],width={'size':12,'offset':0,'order':1},style={'text-align':'center'}),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='dendrogram',figure={}),
                                ],width={'size':12,'offset':0,'order':1}),
                            ]),                        
                        ])
                    ], className='h-100 text-left')
                ], xs=12),
            ], className='p-2 align-items-stretch'),

        ])    
    
    elif active_tab == "table":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6('PCA'),
                    dcc.Input(type="number",id="num_of_pca_6",value=4,min=3,style={'width':'40%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center'}), #,style={'padding-top' : 10}
                dbc.Col([
                    html.H6('Cluster'),
                    dcc.Input(type="number",id="num_of_cluster_6",value=6,min=6,style={'width':'40%','text-align':'center'})
                ],width={'size':1,"offset":0,'order':1},style={'text-align':'center'}),
            ], className='p-2 align-items-stretch'),
            dbc.Row([
                html.H5('Data Table',className='text-center')
            ], className='p-2 align-items-stretch'),
            dbc.Row([
                html.Div(
                    id = 'tableDiv',
                    className = 'tableDiv'),
            ], className='p-2 align-items-stretch'),             
            dbc.Row([
                dcc.Markdown('''
                PCA Table snippet:
                ```python
                n_components=num_of_pca
                pca = PCA(n_components) 
                pca.fit(scaled_data) 
                xpca = pca.transform(scaled_data)
                labels = {str(i): f'PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%' for i in range(n_components)}
                col_list = list(labels.values())

                xpca = np.round(xpca, 2)
                PCA_table = pd.DataFrame(xpca, columns=col_list, index=sample_names)
                PCA_table.insert(0, 'ID', sample_names)
                PCA_table.insert(1, 'Labels', flabels)
                ```''')
            ], className='p-2 align-items-stretch'),       
            dbc.Row([
                dcc.Markdown('''
               Clustering code snippet:
                ```python
                df_clustscores = pd.DataFrame()

                for i in range (2,30):
                  kmeans_m = KMeans(n_clusters=i).fit(data)
                  kmeans_labels = kmeans_m.fit_predict(data)
                  Ksilhouette_avg = silhouette_score(data, kmeans_labels)

                  hclust_m = AgglomerativeClustering(linkage='ward', n_clusters=i).fit(data)
                  hclust_labels = hclust_m.fit_predict(data)
                  Hsilhouette_avg = silhouette_score(data, hclust_labels)

                  birch_m = Birch(n_clusters=i).fit(data)
                  birch_labels = birch_m.fit_predict(data)
                  Bsilhouette_avg = silhouette_score(data, birch_labels) 

                  df_clustscores = df_clustscores.append({'cluster':i,'Kmeans_sa': Ksilhouette_avg, 
                                          'Hclust_sa': Hsilhouette_avg, 
                                          'Birch_sa': Bsilhouette_avg}, 
                                         ignore_index=True)

                  clustscores_long = df_clustscores
                  clustscores_long['cluster'] = clustscores_long.cluster
                  clustscores_long = pd. melt(clustscores_long, id_vars=['cluster'], value_vars=['Kmeans_sa', 'Hclust_sa', 'Birch_sa'])
                  clustscores_long    
                ```''')
            ], className='p-2 align-items-stretch')       

        ])

@app.callback([Output('num_of_x', 'max'),
               Output('num_of_y', 'max'),
               Output('num_of_z', 'max')],
             [Input('num_of_pca', 'value')])
def update_pie_chart(num_of_pca):
    value = num_of_pca - 1
    return value, value, value
    
@app.callback(Output('chart_title', 'children'),
             [Input('radioitems-input', 'value')])
def update_chart_tile(radio_itemns):
    if radio_itemns == '2D':
        return html.Span(f'2D PCA for Labels',style={'text-align':'center'}) 
    elif radio_itemns == '3D':
        return html.Span(f'3D PCA for Labels',style={'text-align':'center'})
    

@app.callback(Output('store-data', 'data'),
             [Input('num_of_pca', 'value')]) 
def update_pie_chart(num_of_pca):
    global PCA_table 
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    labels = {str(i): f'PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%' for i in range(n_components)}
    col_list = list(labels.values())

    xpca = np.round(xpca, 2)
    PCA_table = pd.DataFrame(xpca, columns=col_list, index=sample_names)
    PCA_table.insert(0, 'ID', sample_names)
    PCA_table.insert(1, 'Labels', flabels)
    return PCA_table.to_dict(orient='records')

@app.callback(Output('pie_chart', 'figure'),
             [Input('store-data', 'data'),
              Input('num_of_pca', 'value'),
              Input('radioitems-input', 'value'),
             Input('num_of_x', 'value'),
             Input('num_of_y', 'value'),
             Input('num_of_z', 'value'),
              Input('template_1', 'value')]) 

def update_pie_chart(store,num_of_pca,radio_itemns,num_of_x,num_of_y,num_of_z,template_1):   
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    labels = {str(i): f'PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%' for i in range(n_components)}
    col_list = list(labels.values())
    
    if radio_itemns == '2D':
        figPCA2D = px.scatter(PCA_table,x=col_list[num_of_x], y=col_list[num_of_y], color='Labels', 
                      labels='ID',
                      hover_name = 'ID')
        figPCA2D.update_layout(template=template_1,margin=dict(l=0,r=0,t=0,b=0))
        #figPCA2D.write_image("figPCA.svg")
        #figPCA2D.write_html("figPCA.html")
        return figPCA2D
    if radio_itemns == '3D':              
        figPCA3D = px.scatter_3d(PCA_table,x=col_list[num_of_x], y=col_list[num_of_y], z=col_list[num_of_z],color='Labels', 
                      labels='ID',
                      hover_name = 'ID')
        figPCA3D.update_layout(template=template_1,margin=dict(l=0,r=0,t=0,b=0))
        #figPCA3D.write_image("figPCA.svg")
        #figPCA3D.write_html("figPCA.html")
        return figPCA3D


@app.callback(Output('bar_chart', 'figure'),
             [Input('num_of_pca', 'value'),
              Input('template_1', 'value')])
def update_pie_chart(num_of_pca,template_1):
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    labels = {str(i): f'PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%' for i in range(n_components)}     
    pca_var = pca.explained_variance_ratio_*100
    figPCVar = px.bar(x=labels, y=pca_var)
    figPCVar.update_traces(marker_color='orange')
    figPCVar.update_layout(margin=dict(l=0,r=0,t=0,b=0),template=template_1)
    #figPCVar.write_image("figPCVar.svg")
    #figPCVar.write_html("figPCVar.html")
    return figPCVar   

@app.callback(Output('bar_chart_2', 'figure'),
             [Input('num_of_pca', 'value'),
              Input('template_1', 'value')])
def update_pie_chart(num_of_pca,template_1):
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    figVarCum = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"},
        markers=True
    )
    figVarCum.update_layout(margin=dict(l=0,r=0,t=0,b=0),template=template_1)
    #figVarCum.write_image("figVarCum.svg")
    #figVarCum.write_html("figVarCum.html")

    return figVarCum 
    
@app.callback(Output('pie_chart_2', 'figure'),
             [Input('num_of_pca', 'value'),
              Input('template_1', 'value')])
def update_pie_chart(num_of_pca,template_1):
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    labels = {str(i): f'PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%' for i in range(n_components)}
    total_var = pca.explained_variance_ratio_.sum() * 100

    figPCA_all = px.scatter_matrix(
        xpca,
        color=flabels,
        dimensions=range(n_components),
        labels=labels,
        title=f'Total Explained Variance: {total_var:.2f}%',
    )
    figPCA_all.update_traces(diagonal_visible=False)
    figPCA_all.update_layout(width=1300, height=700, template=template_1)
    #figPCA_all.write_image("figPCA_all.svg")
    #figPCA_all.write_html("figPCA_all.html")
    return figPCA_all   
    
@app.callback(Output('bio_chart', 'figure'),
             [Input('store-data', 'data'),
              Input('num_of_pca', 'value'),
             Input('num_of_x', 'value'),
             Input('num_of_y', 'value'),
              Input('num_of_genes', 'value'),
              Input('overlay', 'value'),
              Input('template_1', 'value')]) 

def update_pie_chart(store,num_of_pca,num_of_x,num_of_y,num_of_genes,overlay,template_1):
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data)
    labels = {str(i): f'PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%' for i in range(n_components)}
    col_list = list(labels.values())
    #clustscores_long = pd.melt(PCA_table, id_vars=['Labels','ID'])
    
    gene_list = data.columns[0:num_of_genes].tolist()
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    figPCBiplot = px.scatter(PCA_table,x=col_list[num_of_x], y=col_list[num_of_y],color='Labels',
                      labels='ID',
                      hover_name = 'ID')
    
    if overlay == 'Overlay':
      for i, feature in enumerate(gene_list):
        figPCBiplot.add_shape(
        type='line',
        line_color='lightgray',
        x0=0, y0=0,
        x1=loadings[i, 0]*4,
        y1=loadings[i, 1]*4
    )
        figPCBiplot.add_annotation(
        x=loadings[i, 0]*4.5,
        y=loadings[i, 1]*4.5,
        ax=1, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
        font=dict(color="gray")
    )
      figPCBiplot.update_layout(template = template_1,clickmode='event')    
      #figPCBiplot.write_image("figPCBiplot.svg")
      #figPCBiplot.write_html("figPCBiplot.html")
      return figPCBiplot
    else:
      figPCBiplot.update_layout(template = template_1,clickmode='event')      
      #figPCBiplot.write_image("figPCBiplot.svg")
      #figPCBiplot.write_html("figPCBiplot.html")
      return figPCBiplot

@app.callback(
    [Output('histogram_name', 'children'),Output('custom_data', 'figure')],
    [Input('bio_chart', 'clickData'),
     Input('template_1', 'value')])
def update_y_timeseries(clickData,template_1):
    country_name = clickData['points'][0]['hovertext']

    dataT = data.transpose()
    fig_histogram = px.histogram(dataT[country_name], nbins=100, marginal="box", color_discrete_sequence=["red","blue"], opacity=0.5)
    fig_histogram.update_layout(showlegend=False,template=template_1)
    #fig_histogram.write_image("fig_histogram.svg")
    #fig_histogram.write_html("fig_histogram.html")
    return html.Span(country_name),fig_histogram

@app.callback([Output('num_of_x_2', 'options'),
               Output('num_of_y_2', 'options'),
               Output('num_of_z_2', 'options')],
             [Input('num_of_pca_2', 'value')])
    
def update_pie_chart(num_of_pca):
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    xpca = np.round(xpca, 2)
    PCA_table = pd.DataFrame(xpca, columns=[f'PC{i}' for i in range (1,n_components+1)])
    PCA_table['Labels'] = flabels
    PCA_table_2 = PCA_table.iloc[:, 0:-1]
    columns_list = list(PCA_table_2.columns)
    return columns_list,columns_list,columns_list
    
@app.callback([Output('pie_chart_3', 'figure'),Output('bar_chart_3', 'figure')],
             [Input('num_of_pca_2', 'value'),
              Input('num_of_cluster_2', 'value'),
             Input('num_of_x_2', 'value'),
             Input('num_of_y_2', 'value'),
             Input('num_of_z_2', 'value'),
              Input('template_2', 'value')])
    
def update_pie_chart(num_of_pca,num_of_cluster,num_of_x,num_of_y,num_of_z,template_2):
    global df_cluster
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    xpca = np.round(xpca, 2)
    PCA_table = pd.DataFrame(xpca, columns=[f'PC{i}' for i in range (1,n_components+1)])
    PCA_table['Labels'] = flabels
    PCA_table_2 = PCA_table.iloc[:, 0:-1]
    n_clusters=num_of_cluster #see above chart for why 4, if 4 Luminal will have 0 samples

    # define and fit the model
    kmeans_m = KMeans(n_clusters).fit(data)
    hclust_m = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters).fit(data)
    birch_m = Birch(n_clusters=n_clusters).fit(data)

    #comapre cluster labels for all methods
    d = {'Labels':flabels, 'Kmeans':kmeans_m.labels_, 'Hclust':hclust_m.labels_,'Birch':birch_m.labels_}
    df_cluster = pd.DataFrame(d)
    df_cluster = pd.merge(df_cluster,PCA_table_2,how='left',left_index=True, right_index=True)
    
    figPCA3D = px.scatter_3d(df_cluster, x=num_of_x, y=num_of_y, 
                         z=num_of_z, 
                         color=df_cluster.Kmeans.astype('category'), 
                         labels=labels)

    figPCA3D.update_layout(margin=dict(l=0,r=0,t=0,b=0),template=template_2)
    #figPCA3D.write_image('figPCA3D_kmeans.svg')
    #figPCA3D.write_html('figPCA3D_kmeans.html')
    crosstab_kmeans = pd.crosstab(df_cluster.Labels, df_cluster.Kmeans, margins=False)
    crosstab_kmeans = crosstab_kmeans.astype('category')

    figKmeans_hm = px.imshow(crosstab_kmeans, aspect='equal', text_auto=True)
    figKmeans_hm.update_layout(margin=dict(l=0,r=0,t=0,b=0),template=template_2)
    #figKmeans_hm.write_image('figKmeans_hm.svg')
    #figKmeans_hm.write_html('figKmeans_hm.html')
    
    return figPCA3D,figKmeans_hm
    
@app.callback(Output('bar_chart_4', 'figure'),
             [Input('range_1', 'value'),
              Input('range_2', 'value'),
              Input('template_2', 'value')])
def update_pie_chart(range_1,range_2,template_2):     
    df_clustscores = pd.DataFrame()

    for i in range (range_1,range_2):
        kmeans_m = KMeans(n_clusters=i).fit(data)
        kmeans_labels = kmeans_m.fit_predict(data)
        Ksilhouette_avg = silhouette_score(data, kmeans_labels)

        df_clustscores = df_clustscores.append({'cluster':i,'Kmeans_sa': Ksilhouette_avg}, ignore_index=True)

    clustscores_long = df_clustscores
    clustscores_long['cluster'] = clustscores_long.cluster
    clustscores_long = pd.melt(clustscores_long, id_vars=['cluster'], value_vars=['Kmeans_sa'])

    fig = px.line(clustscores_long, y='value', x='cluster', color='variable', markers=True)
    fig.update_layout(template=template_2, title_x=0,margin=dict(l=0,r=0,t=0,b=0))
    #fig.write_image('fig_kmeans.svg')
    #fig.write_html('fig_kmeans.html')
    
    return fig
    
@app.callback([Output('num_of_x_3', 'options'),
               Output('num_of_y_3', 'options'),
               Output('num_of_z_3', 'options')],
             [Input('num_of_pca_3', 'value')])
    
def update_pie_chart(num_of_pca):
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    xpca = np.round(xpca, 2)
    PCA_table = pd.DataFrame(xpca, columns=[f'PC{i}' for i in range (1,n_components+1)])
    PCA_table['Labels'] = flabels
    PCA_table_2 = PCA_table.iloc[:, 0:-1]
    columns_list = list(PCA_table_2.columns)
    return columns_list,columns_list,columns_list
    
@app.callback([Output('pie_chart_4', 'figure'),Output('bar_chart_5', 'figure')],
             [Input('num_of_pca_3', 'value'),
              Input('num_of_cluster_3', 'value'),
             Input('num_of_x_3', 'value'),
             Input('num_of_y_3', 'value'),
             Input('num_of_z_3', 'value'),
              Input('template_3', 'value')])
    
def update_pie_chart(num_of_pca,num_of_cluster,num_of_x,num_of_y,num_of_z,template_3):
    global df_cluster_2
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    xpca = np.round(xpca, 2)
    PCA_table = pd.DataFrame(xpca, columns=[f'PC{i}' for i in range (1,n_components+1)])
    PCA_table['Labels'] = flabels
    PCA_table_2 = PCA_table.iloc[:, 0:-1]
    n_clusters=num_of_cluster #see above chart for why 4, if 4 Luminal will have 0 samples

    # define and fit the model
    kmeans_m = KMeans(n_clusters).fit(data)
    hclust_m = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters).fit(data)
    birch_m = Birch(n_clusters=n_clusters).fit(data)

    #comapre cluster labels for all methods
    d = {'Labels':flabels, 'Kmeans':kmeans_m.labels_, 'Hclust':hclust_m.labels_, 
     'Birch':birch_m.labels_}
    df_cluster_2 = pd.DataFrame(d)
    df_cluster_2 = pd.merge(df_cluster_2,PCA_table_2,how='left',left_index=True, right_index=True)
    
    figPCA3D = px.scatter_3d(df_cluster_2, x=num_of_x, y=num_of_y, 
                         z=num_of_z, 
                         color=df_cluster_2.Hclust.astype('category'), 
                         labels=labels)
    figPCA3D.update_layout(margin=dict(l=0,r=0,t=0,b=0),template=template_3) 
    #figPCA3D.write_image('figPCA3D_hclust.svg')
    #figPCA3D.write_html('figPCA3D_hclust.html')
    crosstab_hclust = pd.crosstab(df_cluster_2.Labels, df_cluster_2.Hclust, margins=False)
    crosstab_hclust = crosstab_hclust.astype('category')

    figHclust_hm = px.imshow(crosstab_hclust, aspect='equal', text_auto=True)
    figHclust_hm.update_layout(margin=dict(l=0,r=0,t=0,b=0),template=template_3)
    #figHclust_hm.write_image('figHclust_hm.svg')
    #figHclust_hm.write_html('figHclust_hm.html')
    return figPCA3D,figHclust_hm
    
@app.callback(Output('bar_chart_6', 'figure'),
             [Input('range_3', 'value'),
              Input('range_4', 'value'),
              Input('template_3', 'value')])
def update_pie_chart(range_3,range_4,template_3):     
    df_clustscores = pd.DataFrame()

    for i in range (range_3,range_4):
        hclust_m = AgglomerativeClustering(linkage='ward', n_clusters=i).fit(data)
        hclust_labels = hclust_m.fit_predict(data)
        Hsilhouette_avg = silhouette_score(data, hclust_labels)
        df_clustscores = df_clustscores.append({'cluster':i,'Hclust_sa': Hsilhouette_avg}, ignore_index=True)

    clustscores_long = df_clustscores
    clustscores_long['cluster'] = clustscores_long.cluster
    clustscores_long = pd.melt(clustscores_long, id_vars=['cluster'], value_vars=['Hclust_sa'])

    fig = px.line(clustscores_long, y='value', x='cluster', color='variable', markers=True)
    fig.update_layout(template=template_3, title_x=0,margin=dict(l=0,r=0,t=0,b=0))
    #fig.write_image('fig_hclust.svg')
    #fig.write_html('fig_hclust.html')
    return fig            
            
            
@app.callback([Output('num_of_x_4', 'options'),
               Output('num_of_y_4', 'options'),
               Output('num_of_z_4', 'options')],
             [Input('num_of_pca_4', 'value')])
    
def update_pie_chart(num_of_pca):
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    xpca = np.round(xpca, 2)
    PCA_table = pd.DataFrame(xpca, columns=[f'PC{i}' for i in range (1,n_components+1)])
    PCA_table['Labels'] = flabels
    PCA_table_2 = PCA_table.iloc[:, 0:-1]
    columns_list = list(PCA_table_2.columns)
    return columns_list,columns_list,columns_list
    
@app.callback([Output('pie_chart_5', 'figure'),
               Output('bar_chart_7', 'figure')],
             [Input('num_of_pca_4', 'value'),
              Input('num_of_cluster_4', 'value'),
             Input('num_of_x_4', 'value'),
             Input('num_of_y_4', 'value'),
             Input('num_of_z_4', 'value'),
              Input('template_4', 'value')])
    
def update_pie_chart(num_of_pca,num_of_cluster,num_of_x,num_of_y,num_of_z,template_4):
    global df_cluster_3
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    xpca = np.round(xpca, 2)
    PCA_table = pd.DataFrame(xpca, columns=[f'PC{i}' for i in range (1,n_components+1)])
    PCA_table['Labels'] = flabels
    PCA_table_2 = PCA_table.iloc[:, 0:-1]
    n_clusters=num_of_cluster #see above chart for why 4, if 4 Luminal will have 0 samples

    # define and fit the model
    kmeans_m = KMeans(n_clusters).fit(data)
    hclust_m = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters).fit(data)
    birch_m = Birch(n_clusters=n_clusters).fit(data)

    #comapre cluster labels for all methods
    d = {'Labels':flabels, 'Kmeans':kmeans_m.labels_, 'Hclust':hclust_m.labels_,'Birch':birch_m.labels_}
    df_cluster_3 = pd.DataFrame(d)
    df_cluster_3 = pd.merge(df_cluster_3,PCA_table_2,how='left',left_index=True, right_index=True)
    
    figPCA3D = px.scatter_3d(df_cluster_3, x=num_of_x, y=num_of_y, 
                         z=num_of_z, 
                         color=df_cluster_3.Birch.astype('category'), 
                         labels=labels)
    figPCA3D.update_layout(margin=dict(l=0,r=0,t=0,b=0),template=template_4) 
    #figPCA3D.write_image('figPCA3D_birch.svg')
    #figPCA3D.write_html('figPCA3D_birch.html')
    
    crosstab_birch = pd.crosstab(df_cluster_3.Labels, df_cluster_3.Birch, margins=False)
    crosstab_birch = crosstab_birch.astype('category')

    figBirch_hm = px.imshow(crosstab_birch, aspect='equal', text_auto=True)
    figBirch_hm.update_layout(margin=dict(l=0,r=0,t=0,b=0),template=template_4)
    #figBirch_hm.write_image('figBirch_hm.svg')
    #figBirch_hm.write_html('figBirch_hm.html')

    return figPCA3D,figBirch_hm
    

@app.callback(Output('bar_chart_8', 'figure'),
             [Input('range_5', 'value'),
              Input('range_6', 'value'),
              Input('template_4', 'value')])
def update_pie_chart(range_5,range_6,template_4):     
    df_clustscores = pd.DataFrame()

    for i in range (range_5,range_6):
        birch_m = Birch(n_clusters=i).fit(data)
        birch_labels = birch_m.fit_predict(data)
        Bsilhouette_avg = silhouette_score(data, birch_labels)

        df_clustscores = df_clustscores.append({'cluster':i,'Birch_sa': Bsilhouette_avg}, ignore_index=True)

    clustscores_long = df_clustscores
    clustscores_long['cluster'] = clustscores_long.cluster
    clustscores_long = pd.melt(clustscores_long, id_vars=['cluster'], value_vars=['Birch_sa'])

    fig = px.line(clustscores_long, y='value', x='cluster', color='variable', markers=True)
    fig.update_layout(template=template_4, title_x=0,margin=dict(l=0,r=0,t=0,b=0))
    #fig.write_image('fig_birch.svg')
    #fig.write_html('fig_birch.html')
    return fig     
    
@app.callback(Output('pie_chart_8', 'figure'),
             [Input('range_7', 'value'),
              Input('range_8', 'value'),
              Input('radioitems-input-2', 'value'),
              Input('template_5', 'value')])
def update_pie_chart(range_7,range_8,radioitems_input_2,template_5):     
    global df_clustscores
    df_clustscores = pd.DataFrame()

    for i in range (range_7,range_8):
        kmeans_m = KMeans(n_clusters=i).fit(data)
        kmeans_labels = kmeans_m.fit_predict(data)
        Ksilhouette_avg = silhouette_score(data, kmeans_labels)
        
        hclust_m = AgglomerativeClustering(linkage='ward', n_clusters=i).fit(data)
        hclust_labels = hclust_m.fit_predict(data)
        Hsilhouette_avg = silhouette_score(data, hclust_labels)
        
        birch_m = Birch(n_clusters=i).fit(data)
        birch_labels = birch_m.fit_predict(data)
        Bsilhouette_avg = silhouette_score(data, birch_labels)
        
        if radioitems_input_2 == ['Kmeans_sa']:
            df_clustscores = df_clustscores.append({'cluster':i,'Kmeans_sa': Ksilhouette_avg}, ignore_index=True)
        if radioitems_input_2 == ['Hclust_sa']:
            df_clustscores = df_clustscores.append({'cluster':i, 'Hclust_sa': Hsilhouette_avg}, ignore_index=True)         
        if radioitems_input_2 == ['Birch_sa']:    
            df_clustscores = df_clustscores.append({'cluster':i,'Birch_sa': Bsilhouette_avg}, ignore_index=True)       
        if radioitems_input_2 == ['Kmeans_sa','Hclust_sa']:
            df_clustscores = df_clustscores.append({'cluster':i, 'Kmeans_sa': Ksilhouette_avg,'Hclust_sa': Hsilhouette_avg}, ignore_index=True) 
        if radioitems_input_2 == ['Kmeans_sa','Birch_sa']:
            df_clustscores = df_clustscores.append({'cluster':i, 'Kmeans_sa': Ksilhouette_avg,'Birch_sa': Bsilhouette_avg}, ignore_index=True)
        if radioitems_input_2 == ['Hclust_sa','Birch_sa']:
            df_clustscores = df_clustscores.append({'cluster':i, 'Hclust_sa': Hsilhouette_avg,'Birch_sa': Bsilhouette_avg}, ignore_index=True)
        if radioitems_input_2 == ['Kmeans_sa','Hclust_sa','Birch_sa']:
            df_clustscores = df_clustscores.append({'cluster':i, 'Kmeans_sa': Ksilhouette_avg,'Hclust_sa': Hsilhouette_avg,'Birch_sa': Bsilhouette_avg}, ignore_index=True)
        if radioitems_input_2 == []:
            df_clustscores = df_clustscores.append({'cluster':i, 'Kmeans_sa': Ksilhouette_avg,'Hclust_sa': Hsilhouette_avg,'Birch_sa': Bsilhouette_avg}, ignore_index=True)               
    
    clustscores_long = df_clustscores
    clustscores_long['cluster'] = clustscores_long.cluster
    clustscores_long = pd.melt(clustscores_long, id_vars=['cluster'])

    fig = px.line(clustscores_long, y='value', x='cluster', color='variable', markers=True)
    fig.update_layout(template=template_5, title_x=0,margin=dict(l=0,r=0,t=0,b=0))
    #fig.write_image('fig_all.svg')
    #fig.write_html('fig_all.html')
    return fig
   
    
@app.callback([Output('num_of_x_5', 'options'),
               Output('num_of_y_5', 'options'),
               Output('num_of_z_5', 'options')],
             [Input('num_of_pca_5', 'value')])
    
def update_pie_chart(num_of_pca):
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    xpca = np.round(xpca, 2)
    PCA_table = pd.DataFrame(xpca, columns=[f'PC{i}' for i in range (1,n_components+1)])
    PCA_table['Labels'] = flabels
    PCA_table_2 = PCA_table.iloc[:, 0:-1]
    columns_list = list(PCA_table_2.columns)
    return columns_list,columns_list,columns_list
    
@app.callback(Output('bar_chart_9', 'figure'),
             [Input('num_of_pca_5', 'value'),
              Input('num_of_cluster_5', 'value'),
             Input('num_of_x_5', 'value'),
             Input('num_of_y_5', 'value'),
             Input('num_of_z_5', 'value'),
              Input('template_5', 'value')])    
def update_pie_chart(num_of_pca,num_of_cluster,num_of_x,num_of_y,num_of_z,template_5):
    global df_long
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    xpca = np.round(xpca, 2)
    PCA_table = pd.DataFrame(xpca, columns=[f'PC{i}' for i in range (1,n_components+1)])
    PCA_table['Labels'] = flabels
    PCA_table_2 = PCA_table.iloc[:, 0:-1]
    n_clusters=num_of_cluster #see above chart for why 4, if 4 Luminal will have 0 samples

    # define and fit the model
    kmeans_m = KMeans(n_clusters).fit(data)
    hclust_m = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters).fit(data)
    birch_m = Birch(n_clusters=n_clusters).fit(data)

    #comapre cluster labels for all methods
    d = {'Labels':flabels, 'Kmeans':kmeans_m.labels_, 'Hclust':hclust_m.labels_,'Birch':birch_m.labels_}
    df_cluster = pd.DataFrame(d)
    df_cluster = pd.merge(df_cluster,PCA_table_2,how='left',left_index=True, right_index=True)        

    df_long = df_cluster
    df_long['ID'] = df_long.index
    df_long = pd. melt(df_cluster, id_vars=['ID', num_of_x, num_of_y, num_of_z],value_vars=['Kmeans', 'Hclust', 'Birch'])
    df_long['value'] = df_long['value'].astype('category')
    
    figClusterALL = px.scatter(df_long, x=num_of_x, y=num_of_y, 
                 color='value',
                 facet_col='variable')
    figClusterALL.update_layout(template=template_5,margin=dict(l=0,r=0,t=20,b=0))   
    #figClusterALL.write_image('figClusterALL.svg')
    #figClusterALL.write_html('figClusterALL.html')
    return figClusterALL

@app.callback(Output('bar_chart_10', 'figure'),
             [Input('num_of_pca_5', 'value'),
              Input('num_of_cluster_5', 'value'),
              Input('radioitems-input-2', 'value'),
              Input('template_5', 'value')])    
def update_pie_chart(num_of_pca,num_of_cluster,radioitems_input_2,template_5):
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    xpca = np.round(xpca, 2)
    PCA_table = pd.DataFrame(xpca, columns=[f'PC{i}' for i in range (1,n_components+1)])
    PCA_table['Labels'] = flabels
    PCA_table_2 = PCA_table.iloc[:, 0:-1]
    n_clusters=num_of_cluster #see above chart for why 4, if 4 Luminal will have 0 samples

    # define and fit the model
    kmeans_m = KMeans(n_clusters).fit(data)
    hclust_m = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters).fit(data)
    birch_m = Birch(n_clusters=n_clusters).fit(data)

    #comapre cluster labels for all methods
    d = {'Labels':flabels, 'Kmeans':kmeans_m.labels_, 'Hclust':hclust_m.labels_,'Birch':birch_m.labels_}
    df_cluster = pd.DataFrame(d)
    df_cluster = pd.merge(df_cluster,PCA_table_2,how='left',left_index=True, right_index=True)    
    crosstab_kmeans = pd.crosstab(df_cluster.Labels, df_cluster.Kmeans, margins=False)
    crosstab_kmeans = crosstab_kmeans.astype('category')
    crosstab_hclust = pd.crosstab(df_cluster.Labels, df_cluster.Hclust, margins=False)
    crosstab_hclust = crosstab_hclust.astype('category')
    crosstab_birch = pd.crosstab(df_cluster.Labels, df_cluster.Birch, margins=False)
    crosstab_birch = crosstab_birch.astype('category')
    
    if radioitems_input_2 == ['Kmeans_sa']:
        fig_ClusterHM = make_subplots(rows=1, cols=1,subplot_titles=("Kmeans"),shared_xaxes=True,shared_yaxes=True)
        #fig2 = go.Figure(figBirch_hm.data, figBirch_hm.layout)
        kmeans = go.Heatmap(z=crosstab_kmeans, y=crosstab_kmeans.index, 
                    showscale = False, 
                    text=crosstab_kmeans,
                    texttemplate="%{text}",
                    textfont={"size":10})

        fig_ClusterHM.add_trace(kmeans, row=1, col=1)
        fig_ClusterHM.update_layout(margin=dict(l=0,r=0,t=20,b=0),height=230,template=template_5)
        fig_ClusterHM.update_annotations(font_size=12)        
    if radioitems_input_2 == ['Hclust_sa']:
        fig_ClusterHM = make_subplots(rows=1, cols=1,subplot_titles=("Hclust"),shared_xaxes=True,shared_yaxes=True)
        #fig2 = go.Figure(figBirch_hm.data, figBirch_hm.layout)
        hclust = go.Heatmap(z=crosstab_hclust, y=crosstab_hclust.index, 
                    showscale = False,
                    text=crosstab_hclust,
                    texttemplate="%{text}",
                    textfont={"size":10})

        fig_ClusterHM.add_trace(hclust, row=1, col=1)
        fig_ClusterHM.update_layout(margin=dict(l=0,r=0,t=20,b=0),height=230,template=template_5)    
        fig_ClusterHM.update_annotations(font_size=12)    
    if radioitems_input_2 == ['Birch_sa']:
        fig_ClusterHM = make_subplots(rows=1, cols=1,subplot_titles=("Birch"),shared_xaxes=True,shared_yaxes=True) #,shared_xaxes=True,shared_yaxes=True
        #fig2 = go.Figure(figBirch_hm.data, figBirch_hm.layout)
        birch = go.Heatmap(z=crosstab_birch, y=crosstab_birch.index,
                    showscale = False, 
                    text=crosstab_birch,
                    texttemplate="%{text}",
                    textfont={"size":10})

        fig_ClusterHM.add_trace(birch, row=1, col=1)    
        fig_ClusterHM.update_layout(margin=dict(l=0,r=0,t=20,b=0),height=230,template=template_5)    
        fig_ClusterHM.update_annotations(font_size=12)    
    if radioitems_input_2 == ['Kmeans_sa','Hclust_sa']:
        fig_ClusterHM = make_subplots(rows=1, cols=2,subplot_titles=("Kmeans","Hclust"),shared_xaxes=True,shared_yaxes=True)
        #fig2 = go.Figure(figBirch_hm.data, figBirch_hm.layout)
        kmeans = go.Heatmap(z=crosstab_kmeans, y=crosstab_kmeans.index, 
                    showscale = False, 
                    text=crosstab_kmeans,
                    texttemplate="%{text}",
                    textfont={"size":10})
        hclust = go.Heatmap(z=crosstab_hclust, y=crosstab_hclust.index, 
                    showscale = False,
                    text=crosstab_hclust,
                    texttemplate="%{text}",
                    textfont={"size":10})
        
        fig_ClusterHM.add_trace(kmeans, row=1, col=1)
        fig_ClusterHM.add_trace(hclust, row=1, col=2)
        fig_ClusterHM.update_layout(margin=dict(l=0,r=0,t=20,b=0),height=230,template=template_5)
        fig_ClusterHM.update_annotations(font_size=12)   
    if radioitems_input_2 == ['Kmeans_sa','Birch_sa']:
        fig_ClusterHM = make_subplots(rows=1, cols=2,subplot_titles=("Kmeans","Birch"),shared_xaxes=True,shared_yaxes=True)
        #fig2 = go.Figure(figBirch_hm.data, figBirch_hm.layout)
        kmeans = go.Heatmap(z=crosstab_kmeans, y=crosstab_kmeans.index, 
                    showscale = False, 
                    text=crosstab_kmeans,
                    texttemplate="%{text}",
                    textfont={"size":10})
        birch = go.Heatmap(z=crosstab_birch, y=crosstab_birch.index,
                    showscale = False, 
                    text=crosstab_birch,
                    texttemplate="%{text}",
                    textfont={"size":10})
        
        fig_ClusterHM.add_trace(kmeans, row=1, col=1)
        fig_ClusterHM.add_trace(birch, row=1, col=2)
        fig_ClusterHM.update_layout(margin=dict(l=0,r=0,t=20,b=0),height=230,template=template_5)    
        fig_ClusterHM.update_annotations(font_size=12)   
    if radioitems_input_2 == ['Hclust_sa','Birch_sa']:
        fig_ClusterHM = make_subplots(rows=1, cols=2,subplot_titles=("Hclust","Birch"),shared_xaxes=True,shared_yaxes=True)
        hclust = go.Heatmap(z=crosstab_hclust, y=crosstab_hclust.index, 
                    showscale = False,
                    text=crosstab_hclust,
                    texttemplate="%{text}",
                    textfont={"size":10})
        birch = go.Heatmap(z=crosstab_birch, y=crosstab_birch.index,
                    showscale = False, 
                    text=crosstab_birch,
                    texttemplate="%{text}",
                    textfont={"size":10})        
        
        fig_ClusterHM.add_trace(hclust, row=1, col=1)
        fig_ClusterHM.add_trace(birch, row=1, col=2)     
        fig_ClusterHM.update_layout(margin=dict(l=0,r=0,t=20,b=0),height=230,template=template_5)
        fig_ClusterHM.update_annotations(font_size=12)    
    if radioitems_input_2 == ['Kmeans_sa','Hclust_sa','Birch_sa']:
        fig_ClusterHM = make_subplots(rows=1, cols=3,subplot_titles=("Kmeans","Hclust","Birch"),shared_xaxes=True,shared_yaxes=True)
        kmeans = go.Heatmap(z=crosstab_kmeans, y=crosstab_kmeans.index, 
                    showscale = False, 
                    text=crosstab_kmeans,
                    texttemplate="%{text}",
                    textfont={"size":10})               
        hclust = go.Heatmap(z=crosstab_hclust, y=crosstab_hclust.index, 
                    showscale = False,
                    text=crosstab_hclust,
                    texttemplate="%{text}",
                    textfont={"size":10})
        birch = go.Heatmap(z=crosstab_birch, y=crosstab_birch.index,
                    showscale = False, 
                    text=crosstab_birch,
                    texttemplate="%{text}",
                    textfont={"size":10})        
        fig_ClusterHM.add_trace(kmeans, row=1, col=1)       
        fig_ClusterHM.add_trace(hclust, row=1, col=2)
        fig_ClusterHM.add_trace(birch, row=1, col=3)     
        fig_ClusterHM.update_layout(margin=dict(l=0,r=0,t=20,b=0),height=230,template=template_5)
        fig_ClusterHM.update_annotations(font_size=12)
    
    if radioitems_input_2 == []:
        fig_ClusterHM = make_subplots(rows=1, cols=3,subplot_titles=("Kmeans","Hclust","Birch"),shared_xaxes=True,shared_yaxes=True)
        kmeans = go.Heatmap(z=crosstab_kmeans, y=crosstab_kmeans.index, 
                    showscale = False, 
                    text=crosstab_kmeans,
                    texttemplate="%{text}",
                    textfont={"size":10})               
        hclust = go.Heatmap(z=crosstab_hclust, y=crosstab_hclust.index, 
                    showscale = False,
                    text=crosstab_hclust,
                    texttemplate="%{text}",
                    textfont={"size":10})
        birch = go.Heatmap(z=crosstab_birch, y=crosstab_birch.index,
                    showscale = False, 
                    text=crosstab_birch,
                    texttemplate="%{text}",
                    textfont={"size":10})        
        fig_ClusterHM.add_trace(kmeans, row=1, col=1)       
        fig_ClusterHM.add_trace(hclust, row=1, col=2)
        fig_ClusterHM.add_trace(birch, row=1, col=3)     
        fig_ClusterHM.update_layout(margin=dict(l=0,r=0,t=20,b=0),height=230,template=template_5)       
        fig_ClusterHM.update_annotations(font_size=12)
    #fig_ClusterHM.write_image('fig_ClusterHML.svg')
    #fig_ClusterHM.write_html('fig_ClusterHM.html')
    return fig_ClusterHM    

@app.callback(Output('dendrogram', 'figure'),
             [Input('num_of_pca_5', 'value'),
              Input('template_5', 'value')])    
def update_pie_chart(num_of_pca,template_5):
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    figDendrogram = ff.create_dendrogram(xpca, color_threshold=4.5, labels=flabels)
    figDendrogram.update_layout(template=template_5)
    #figDendrogram.write_image('figDendrogram.svg')
    #figDendrogram.write_html('figDendrogram.html')
    return figDendrogram

@app.callback(Output('tableDiv','children'),
             [Input('num_of_pca_6', 'value'),
              Input('num_of_cluster_6', 'value')])

def update_data_2(num_of_pca,num_of_cluster):
    n_components=num_of_pca
    pca = PCA(n_components) 
    pca.fit(scaled_data) 
    xpca = pca.transform(scaled_data)
    labels = {str(i): f'PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%' for i in range(n_components)}
    col_list = list(labels.values())

    xpca = np.round(xpca, 2)
    PCA_table = pd.DataFrame(xpca, columns=col_list, index=sample_names)
    PCA_table.insert(0, 'ID', sample_names)
    PCA_table.insert(1, 'Labels', flabels) 
    PCA_table = PCA_table.reset_index(drop=True)
    n_clusters=num_of_cluster #see above chart for why 4, if 4 Luminal will have 0 samples

    # define and fit the model
    kmeans_m = KMeans(n_clusters).fit(data)
    hclust_m = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters).fit(data)
    birch_m = Birch(n_clusters=n_clusters).fit(data)

    #comapre cluster labels for all methods
    d = {'Kmeans':kmeans_m.labels_, 'Hclust':hclust_m.labels_,'Birch':birch_m.labels_}
    df_cluster = pd.DataFrame(d)
    df_cluster = pd.merge(df_cluster,PCA_table,how='left',left_index=True, right_index=True)  
    mycolumns = [{'name': i, 'id': i} for i in df_cluster.columns]
    
    return html.Div([
            dash_table.DataTable(
            id='table',
            columns=mycolumns,
            data=df_cluster.to_dict("rows"),
            style_table={'overflow':'scroll','height':550},
            style_header={'backgroundColor':'orange','padding':'10px','color':'#000000'},
            style_cell={'textAlign':'center','font_size': '12px',
                       'whiteSpace':'normal','height':'auto'},
            editable=True,              # allow editing of data inside all cells
            filter_action="native",     # allow filtering of data by user ('native') or not ('none')
            sort_action="native",       # enables data to be sorted per-column by user or not ('none')
            sort_mode="single",         # sort across 'multi' or 'single' columns
            column_selectable="multi",  # allow users to select 'multi' or 'single' columns
            row_selectable="multi",     # allow users to select 'multi' or 'single' rows
            row_deletable=True,         # choose if user can delete a row (True) or not (False)
            selected_columns=[],        # ids of columns that user selects
            selected_rows=[],           # indices of rows that user selects
            page_action="native")
    
    ])


if __name__ == '__main__':
    app.run_server(debug=False,host="0.0.0.0",port=8080)