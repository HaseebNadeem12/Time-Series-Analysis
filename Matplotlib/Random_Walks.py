# """Use of choise function"""
# from random import choice
# a=0
# while a<3:
#     a +=1
#     x_values = [0]
#     y_values = [0]
#
#     x_direction = choice([-1, 1])
#     print(x_direction)
#
#     x_distance = choice([0, 1, 2, 3, 4])
#     print(x_distance )
#
#     x_step = x_direction * x_distance
#     print(x_step,'\n')
#
#     y_direction = choice([-1, 1])
#     print(y_direction)
#
#     y_distance = choice([0, 1, 2, 3, 4])
#     print(y_distance)
#
#     y_step = y_direction * y_distance
#     print(y_step, '\n')
#
#     # Calculate the next x and y values
#     next_x = x_values[-1] + x_step
#     print(next_x)
#
#     next_y = y_values[-1] + y_step
#     print(next_y , '\n')
#
#     x_values.append(next_x)
#     y_values.append(next_y)


"""============================================================"""
from random import choice
import matplotlib.pyplot as plt

# A class to generate a random walk
class RandomWalk:
    def __init__(self, num_points):
        """Initialize attributes for a walk"""
        self.num_points = num_points

        # All walks start at (0, 0)
        self.x_values = [0]
        self.y_values = [0]

    def fill_walk(self):
        """Calculate all the points in the walk"""
        while len(self.x_values) < self.num_points:

            # Deciding which direction to go & how far to go
            x_direction = choice([-1, 1])
            x_distance = choice([0, 1, 2, 3, 4])
            x_step = x_direction * x_distance

            y_direction = choice([-1, 1])
            y_distance = choice([0, 1, 2, 3, 4])
            y_step = y_direction * y_distance

            # Reject moves that go nowhere
            if x_step == 0 and y_step == 0:
                continue

            # Calculate the next x and y values
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step

            self.x_values.append(next_x)
            self.y_values.append(next_y)

        # Creating an instance of RandomWalk
        while True:
            R_walk = RandomWalk(5000)
            R_walk.fill_walk()

            # Set the size of the plotting window
            plt.figure(dpi=128, figsize=(10, 6))

            # Plotting the points in the walk as a line plot
            plt.plot(R_walk.x_values, R_walk.y_values, linewidth=1)

            # Highlight the start and end points
            plt.scatter(0, 0, c='green', edgecolors='none', s=100)
            plt.scatter(R_walk.x_values[-1], R_walk.y_values[-1], c='red', edgecolors='none', s=100)

            # Remove the axes
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)

            plt.show()

            keep_running = input('Want to make another Random Walk, enter Yes or NO: ')
            if keep_running.lower() == 'no':
                break