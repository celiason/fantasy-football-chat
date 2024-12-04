import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Polygon

def chull_plot(df, alpha=0.25, palette='pastel'):
    """
    
    df: a pandas data frame object with columns x, y, and grp
    
    Example:
    df = X_pca
    df['label'] = labels
    df.columns = ['x','y','grp']
    
    chull_plot(df, palette='Spectral', alpha=0.25)
    
    """
    
    groups = df['grp']

    # Add colors
    pal = sns.color_palette(palette, len(groups.unique()))

    # Calculate the convex hulls for each group
    hulls = []
    for group in groups.unique():
        ss = df[df['grp']==group]
        if len(ss) < 3:
            hull = Polygon()
        else:
            hull = ConvexHull(ss[['x', 'y']].values)
            points = MultiPoint(ss[['x', 'y']].values)
            hull = points.convex_hull
        
        hulls.append(hull)

    plt.figure(figsize=(12,8))

    # Plot the convex hulls
    for i in range(len(hulls)):
    # for hull in hulls:
        hull = hulls[i]
        # plt.plot(*hull.exterior.xy)
        plt.fill(*hull.exterior.xy, color=pal[i], alpha=alpha)

    # Add the scatter plot
    sns.scatterplot(data=df, x='x', y='y', hue='grp', palette=pal)

    plt.show()

