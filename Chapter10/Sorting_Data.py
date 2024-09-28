# Recalling users
import json

def get_stored_username():
    """Get stored username if available"""
    file_name = 'user_name.txt'

    try:
        with open(file_name,'r') as file2:
            name = json.load(file2)
    except FileNotFoundError:
        return None
    else:
        return name

def get_new_user():
    file_name = 'user_name.txt'
    name = input('What is your name: ')
    with open(file_name, 'w') as file1:
        json.dump(name, file1)
        return name
def greet_user():
    """Greet the user name"""
    userName = get_stored_username()
    if userName:
        print("Is " + userName + " is your correct username...?")
        answer = input("Enter YES or NO: ")

        if answer.lower() == 'yes':
            print("Welcome back " + userName)

        else:
            name3 = get_new_user()
            print("We will remember your name " + name3)

    else:
        userName_1 = get_new_user()
        print("We will remember your name " + userName_1 )


print(greet_user())