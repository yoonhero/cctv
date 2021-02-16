import json
import requests
KAKAO_TOKEN = "q5BQ4c4yllEjuQX8Lc0Nw6aIHIbO4phB_8ycZgorDNQAAAF3qrHYfw"
url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

# 사용자 토큰
headers = {
    "Authorization": "Bearer " + KAKAO_TOKEN
}

data = {
    "template_object": json.dumps({"object_type": "text",
                                   "text": "Hello, world!",
                                   "link": {
                                        "web_url": "www.google.com",
                                        "mobile_web_url": "www.google.com"
                                   }
})
}

response = requests.post(url, headers=headers, data=data)
print(response.status_code)

if response.json().get('result_code') == 0:
    print('메시지를 성공적으로 보냈습니다.')
else:
    print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))




