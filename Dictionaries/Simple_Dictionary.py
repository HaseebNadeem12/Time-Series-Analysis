"""Simple Dictionary"""
alien_0 = {'my_name':'Haseeb',
           'Age':23,
           'Birth_Year':'12-07-2001',}
# print(alien_0)
# print(alien_0.keys())
# print(alien_0.values())

"""Adding New Key Value Pair"""
# alien_0['New_Variable'] = 'My Full Name'
# print(alien_0)

"""Starting With an empty Dictionary"""
# my_dictionary = {}
# my_dictionary['Your Name'] = 'Haseeb Nadeem'
# my_dictionary['Father Name'] = 'Nadeem Umer'
# print(my_dictionary)

"""Modifying Values"""
# my_dictionary['Your Name'] = 'Fahad Nameed'
# print(my_dictionary)

"""Removing Key Value Pair"""
# del my_dictionary['Your Name']
# print(my_dictionary)

"""A Dictionary of Similar Objects"""
# Favorite_Language = {'Ali':'C++',
#                      'Haseeb':'Python',
#                      'Sara':'java script',
#                      'Amena':'R',}
# print("Sara's favorite language is " + Favorite_Language['Sara'].title())
#
"""Looping Through a Dictionary Key """
# Favorite_Language = {'Ali':'C++',
#                      'Haseeb':'Python',
#                      'Sara':'Java Script',
#                      'Amena':'R',}
#
# for key in Favorite_Language.keys():
#     print(key)
#
# for val in Favorite_Language.values():
#     print(val)
#
# for key in Favorite_Language.keys():
#     print(Favorite_Language[key])
#
# for key, value, in Favorite_Language.items():
#     print(key  , value)

"""Looping Through A Dictionary"""
# Favorite_Language = {'Ali':'C++',
#                      'Haseeb':'Python',
#                      'Sara':'Java Script',
#                      'Amena':'R',}
#
# for name, Language in Favorite_Language.items():
#
#     print(name.title() + "'s favorite language is " + Language.title())

# """Example"""
Favorite_Language = {'Ali':'C++',
                     'Haseeb':'Python',
                     'Sara':'Java Script',
                     'Amena':'C#',
                     'Hashir':'Larawell',
                     'Daniyal':'Python',
                     'Zubair':'C#',}

List_of_Favorite_language = ['Asim','Amena','Haseeb','Daniyal','Asma']

#-> If name matches, print we know your favorite language

# for key in Favorite_Language.keys():
#
#     # print(key.title())
#
#     if(key in List_of_Favorite_language):
#         # print(key)
#         print("\tHi "+ key.title() + " we know your favorite language is " + Favorite_Language[key])

# -> in order(Sorted)
# for key in sorted(Favorite_Language.keys()):
#
#     # print(key.title())
#
#     if (key in List_of_Favorite_language):
#         print(key.title())
#         print("\tHi " + key.title() + " we know your favorite language is " + Favorite_Language[key])
#

#-> include unique value(no repitation)
# print("Following all languas are mention")
#
# for values in set(Favorite_Language.values()):
#
#     print(values)

"""A List of Dictionary"""
# alien_0 = {'colour': 'green' , 'points': 5 ,}
# alien_1 = {'colour': 'yellow' , 'points': 10 ,}
# alien_2= {'colour': 'red' , 'points': 15 ,}
#
# aliens = [alien_0,alien_1,alien_2]
#
# for alien in aliens:
#     print(alien)


"""Automatically Genrating Alien List"""
# list_of_aliens = []
#
# for alien_number in range(7):
#
#     my_input01 = input("You want to run the loop, enter Yes/No: ")
#     if my_input01.lower() == "yes":
#         new_alien = {'colour': 'green', 'points': 5, 'speed': 'slow'}
#         list_of_aliens.append(new_alien)
#
# for alien_number in range(8,14):
#
#     new_alien = {'colour': 'yellow' , 'points': 10 , 'speed': 'medium'}
#     list_of_aliens.append(new_alien)
#
# for alien_number in range(14,21):
#
#     new_alien = {'colour': 'red' , 'points': 10 , 'speed': 'medium'}
#     list_of_aliens.append(new_alien)
#
#
# for list_of_alien in list_of_aliens[:21]:
#     print(list_of_alien)

# print("Total number of alien " + str(len(list_of_aliens)))
#
# # To change the colour of all alien
# for list_of_alien in list_of_aliens[0:]:
#
#     if list_of_alien['colour'] == 'green':
#         list_of_alien['colour'] = 'red'
#         list_of_alien['points'] = 15
#         list_of_alien['speed'] = 'fast'
#
#     elif list_of_alien['colour'] == 'yellow':
#         list_of_alien['colour'] = 'green'
#         list_of_alien['points'] = 5
#         list_of_alien['speed'] = 'slow'
#
#     elif list_of_alien['colour'] == 'red':
#         list_of_alien['colour'] = 'yellow'
#         list_of_alien['points'] = 10
#         list_of_alien['speed'] = 'medium'
#
# for list_of_alien in list_of_aliens[0:]:
#     print(list_of_alien)



user_dic = {}
your_name = input("Enter your name: ")
user_dic['name']











