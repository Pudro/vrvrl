import matplotlib.pyplot as plt

def generate_route_plot(locations, routes) -> plt.Figure:
    fig, ax = plt.subplots()

    x, y = zip(*locations)

    ax.scatter(x, y, color='red', marker='o')

    route_colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink', 'black', 'yellow', 'grey', 'silver', 'magenta', 'brown', 'indigo']

    for index, route in enumerate(routes):
        color = route_colors[index % len(route_colors)]

        for i in range(len(route) - 1):
            ax.plot([x[route[i]], x[route[i + 1]]], [y[route[i]], y[route[i + 1]]], color=color)

        ax.plot([x[route[-1]], x[route[0]]], [y[route[-1]], y[route[0]]], color='blue')

    return fig
