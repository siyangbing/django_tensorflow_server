import os

ip_address = "0.0.0.0:8080"

commond_str = "python manage.py runserver  {}".format(ip_address)

os.system(commond_str)
