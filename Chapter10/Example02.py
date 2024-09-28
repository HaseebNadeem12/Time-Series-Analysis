# # 10-3
# name1 = input('Wnter your name: ')
#
# with open('name.txt','w') as file_name:
#     file_name.write(name1)


## 10-4

# with open('guest.txt','w') as guest_book:
#
#     while True:
#
#         names = input('Enter your name or(Type exit to quite) ')
#
#         if names.lower() == 'excit':
#             break
#
#         else:
#             print(f"Hello {names} ! welcome ")
#
#         guest_book.write(f"{names} just visited our guest_book: \n")
#
# print("ThankYou for choosing us! :) ")


## 10-5

with open('guest.txt','a') as guest_book:

    while True:

        name = input('Enter your name: ')
        reason = input('\nEnter your reason WHY YOU LIKE PROGRAMMING (Type exit to quite): ')

        if reason.lower() == 'excit':
            break

        else:
            print(f"Hello {name} ! welcome ")

            guest_book.write(reason)
            print(f"{name} we just stored your response in our guest_book: \n")

print("ThankYou for choosing us! :) ")