from urllib import response
import requests

url = 'http://127.0.0.1:8000/sentence/'
response = requests.post(url, json={'sentence': 'tài liệu hay quá'})
print(response.text)
