import pandas as pd
import numpy as np
from plotly.graph_objs import graph_objs
import plotly.figure_factory as ff
import plotly.offline as py

train = pd.read_csv('./data/iris.csv')
train = train.drop(['Id'], 1)

print(train.head())

sx = train['SepalLengthCm']
sy = train['SepalWidthCm']
px = train['PetalLengthCm']
py = train['PetalWidthCm']

train['Species'] = train['Species'].apply(lambda x: x.replace('-', ''))
cats = train['Species'].unique()
print(cats[1])

# trace = go.Scatter3d(
#   x=x,
#   y=y,
#   z=z,
#   mode='markers',
#   marker=dict(
#     size=12,
#     line=dict(
#       color='rgba(217, 217, 217, 0.14)',
#       width=0.5
#     ),
#     opacity=0.8
#   )
# )

# data = [trace]

# layout = go.Layout(

# )

# fig = go.Figure(data=data, layout=layout)

# py.plot(fig, '3dscatter.html')

fig = ff.create_scatterplotmatrix(train, diag='box', index='Species',
  colormap= dict(
    Irissetosa = '#00F5FF',
    Irisversicolor = '#32CD32',
    Irisvirginica = '#DAA520'
  ),
 colormap_type='cat', height=800, width=800)
py.plot(fig, filename='Scatter')