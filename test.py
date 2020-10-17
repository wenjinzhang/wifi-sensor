import requests
dict1 = {'start_time': '14:52:27', 'end_time': '', 'inference_time': '', 'act': '', 'user': ''}
dict2 = {'start_time': '14:52:27', 'end_time': '14:52:31', 'inference_time': 0.08132, 'act': 'squating', 'user': 'Wenjin'}

url = 'http://127.0.0.1:5000/activity'

r = requests.post(url, data=dict2)


