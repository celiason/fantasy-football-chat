# Create a "bump chart" of rankings with smoothed sigmoid-like transitions

import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np

query1 = """
with cte as (
  select *,
  case  when source_manager_id is null or source_manager_id = destination_manager_id then destination_manager_id
        when destination_manager_id is null or source_manager_id = destination_manager_id then source_manager_id
        when source_manager_id != destination_manager_id then destination_manager_id
      else NULL
    end as manager_id
from events where type != 'trade'
)

select year, m.manager,
  sum(case when type = 'add' then 1 else 0 end) as adds,
  sum(case when type = 'drop' then 1 else 0 end) as drops,
  sum(case when type in ('active', 'inactive') then 1 else 0 end) as rosters
from cte
left join managers m
  on cte.manager_id = m.manager_id
group by year, m.manager
order by year, m.manager
;
"""

from langchain_community.utilities import SQLDatabase

def init_database(password: str, database: str) -> SQLDatabase:
  db_uri = f"postgresql://postgres.rpeohwliutyvtvmcwkwh:{password}@aws-0-us-west-1.pooler.supabase.com:6543/{database}"
  # Here I'm limiting the LLM to only 2 tables (makes things easier, gives better results)
  return SQLDatabase.from_uri(db_uri, include_tables = ['slots', 'standings', 'team_names'], view_support=True)

db = init_database('slow_learners_2007', 'postgres')
# print(db.run(query1))
# print(db.run("select * from standings limit 10;"))
# pd.read_sql(query1, engine)
df = pd.read_csv("/Users/chad/Downloads/supabase_rpeohwliutyvtvmcwkwh_Standings View.csv")
# df2 = pd.read_csv("/Users/chad/Downloads/supabase_rpeohwliutyvtvmcwkwh_Standings Table.csv")
# df = df.merge(df2, on=['year','manager'])


def bumpchart(df):

    """
    df = data frame with the following columns: year, rank, manager/group
    """

    # Color scheme

    # Create a pivot table
    df_ranks = df[['year','rank','manager']].pivot(index='year', columns='manager', values='rank')
    df_ranks.reset_index(inplace=True)

    fig = px.line(df_ranks, x='year', y=df_ranks.columns[1:], markers=True, title='Bump Chart', color_discrete_sequence=px.colors.qualitative.Antique, line_shape='spline')
    
    # Get list of unique managers
    managers = sorted(df['manager'].unique().tolist())

    # Create sigmoid curve for smooth transitions
    smooth_data = []
    for manager in managers:
        cat_data = df[df['manager'] == manager]
        for i in range(len(cat_data) - 1):
            t1, r1 = cat_data.iloc[i][['year', 'rank']]
            t2, r2 = cat_data.iloc[i + 1][['year', 'rank']]
            times = np.linspace(t1, t2, 50)  # 50 points for smooth curve
            ranks = r1 + (r2 - r1) / (1 + np.exp(-10 * (times - (t1 + t2) / 2)))
            smooth_data.append(pd.DataFrame({'year': times, 'rank': ranks, 'manager': manager}))

    # Concatenate the smoothed data
    smooth_df = pd.concat(smooth_data)

    # Create the line plot with Plotly Express
    fig = px.line(smooth_df, x='year', y='rank', color='manager',
                labels={'rank': 'Rank', 'year': 'Year'},
                title='Bump Chart showing Team Ranks Over Time',
                color_discrete_sequence=px.colors.qualitative.Pastel_r,
                line_shape='spline'
                )
    
    # Change line thickness
    fig.update_traces(line=dict(width=2.5))
    
    # Remove hover info for the lines
    fig.update_traces(hoverinfo='skip', hovertemplate=None)

    # Extract the line colors from Plotly Express
    line_colors = {trace['name']: trace['line']['color'] for trace in fig['data'] if 'line' in trace}
    
    # Add custom colors for managers without name in line_colors
    idx = set(df['manager']) - line_colors.keys()
    for manager in idx:
        line_colors[manager] = 'gray'

    # Add markers combined under the same legend group
    for manager in managers:
        cat_data = df[df['manager'] == manager]
        fig.add_trace(go.Scatter(
            x=cat_data['year'],
            y=cat_data['rank'],
            mode='markers',
            marker=dict(size=10, color=line_colors[manager]),
            name=manager,          # Use the same name as the line
            hovertemplate='<b>%{text}</b>: #%{y} in %{x}',
            text=cat_data['manager'],
            legendgroup=manager,   # Group legend items
            showlegend=False        # Hide legend for markers
        ))

    fig.update_layout(
        plot_bgcolor='rgb(80,80,80)',
        paper_bgcolor='rgb(80,80,80)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        width=1200,
        height=600
    )

    # Show all years on x axis
    fig.update_xaxes(tickvals=df['year'].unique())

    # Show rank as 1st, 2nd, 3rd, etc.
    fig.update_yaxes(tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                     ticktext=['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th'])
    
    # Orient x axis labels quasi-vertically
    fig.update_xaxes(tickangle=0)

    # Set fixed ranges when different traces are shown
    fig.update_layout(yaxis=dict(range=[12.5, 0.5], showgrid=False, fixedrange=True),
                      xaxis=dict(range=[2006.5, 2023.5], showgrid=False, fixedrange=True))

    # fig.update_xaxes(rangeslider_visible=False)

    return fig


fig = bumpchart(df)

fig.write_html("bumpchart.html")

