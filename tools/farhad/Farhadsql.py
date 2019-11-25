from  mysql import connector
from farhad.Farhadcolor import tcolors,bcolors


mydb = connector.connect(
    
    password='shad1190102528',
    host='127.0.0.1',
    user='root'
    #,database='project'
    #,auth_plugin='mysql_native_password'
    )
    
print(bcolors.BLUE)
print("Farhad, you are conected to your Sql")
print(tcolors.ENDC)