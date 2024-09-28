# # Create a file named 'message_2.txt' and write to it
# with open('message_2.txt', 'w') as file_2:
#     file_2.write('In Python, you can do so many things\n')
#     file_2.write('In Python, we can also do machine learning and deep learning\n')
#     file_2.write('In Python, we can also do web development\n')
#     file_2.write('In Python, we can also make chatbots\n')
#
# # Read the file and print its content
# with open('message_2.txt', 'r') as file_3:
#     file1 = file_3.readlines()
#     print(file1)
#
# # Print each line
# for line in file1:
#     print(line)
#
# # for line in file1:
# #     new_lines = [line.replace('Python','java')]
#
# new_lines = [line.replace('Python', 'java') for line in file1]
#
# with open('message_2.txt', 'w') as my_file:
#     my_file.writelines(new_lines)
"""-----------------------------------------------------------------"""

# # Analysing text
#
# file_name = 'name.txt'
#
# try:
#     with open(file_name,'r') as file:
#         contents = file.read()
#
# except FileNotFoundError:
#     print("Sorry the file " + file_name + " not exist: ")
#
# else:
#     words = contents.strip("")
#     words_length = len(words)
#     print("The file " + file_name + " has about " + str(words_length) + " words: ")


"""-----------------------------------------------------------------"""
# while True:
#
#     print("Enter any two numbers (or type QUIT to excit)")
#     try:
#         num1 = int(input("\t* "))
#         num2 = int(input("\t* "))
#
#     except ValueError:
#         print("Please make sure you enter only numbers: ")
#
#     else:
#         add = num1 + num2
#
#         print("Result of addition is : " + str(add))


"""-----------------------------------------------------------------"""
# # counting perticular word
# with open('name.txt','r') as my_name:
#     names = my_name.read()
#     pos = names.count('low')
#     print(pos)