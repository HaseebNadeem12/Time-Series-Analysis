# import matplotlib.pyplot as plt

# # Sample data
# x = [1,2,3,4,5]
# y = [1,4,9,16,25]
#
# # plotting graph
# plt.scatter(x,y,linewidth=3)
#
# # add lables and titles
# plt.xlabel("X-axis",fontsize=14)
# plt.ylabel("Y-axis",fontsize=14)
# plt.title("Simple line plot",fontsize=14)
#
"""------------------------------------------------------------"""
# # show plot
# import matplotlib.pyplot as plt
# plt.show()

# x_axis = list(range(0,2500))
# y_axis = [x**2 for x in x_axis]
#
# # plt.plot(x_axis,y_axis,c='red',edgecolors='none', s=40)
# # plt.scatter(x_axis,y_axis,c='red',edgecolors='none', s=40)
# # plt.scatter(x_axis,y_axis,c=(0,0,0.8),edgecolors='none', s=40)
# plt.scatter(x_axis,y_axis,c=y_axis, cmap=plt.cm.Reds ,edgecolors='none', s=40)
#
#
# plt.axis([0,1100,0,1100000])
#
# # plt.show()
# plt.savefig('squares_plot.png',bbox_inches = 'tight')


"""------------------------------------------------------------"""
# sample plot for power cube

# import matplotlib.pyplot as plt
#
# x_value = [1,3,5,7,9]
# y_value = [2,4,6,8,10]
#
# plt.xlabel("X-axis",fontsize = 12)
# plt.ylabel("Y-axis",fontsize = 12)
# plt.title('Example',fontsize=12)
#
# plt.scatter(x_value,y_value,linewidths=5)
#
# plt.show()

# import matplotlib.pyplot as plt
#
# x_value = list(range(0,501))
# y_value = [x**3 for x in x_value]
#
# plt.xlabel("X-axis",fontsize = 12)
# plt.ylabel("Y-axis",fontsize = 12)
# plt.title('Example',fontsize=12)
#
# # plt.scatter(x_value,y_value,linewidths=5)
# # plt.scatter(x_value,y_value, c='red' ,linewidths=5)
# plt.scatter(x_value,y_value, c=x_value, cmap = plt.cm.Blues ,linewidths=5)
#
# plt.show()