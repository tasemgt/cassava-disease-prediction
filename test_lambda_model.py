import requests

url = "https://47ub54xazq3kgshw3al2tghuye0bsomz.lambda-url.eu-west-2.on.aws/"

# With JSON payload
payload = {
    "url": "https://thebluefufu.com/wp-content/uploads/2022/04/Enjoy-Your-Life-1080-%C3%97-1080-piksel.png"
}

response = requests.post(url, json=payload)
print(response.json())

# Or with base64 encoded image
# import base64

# with open("image.jpg", "rb") as f:
#     image_data = base64.b64encode(f.read()).decode('utf-8')

# payload = {
#     "image": image_data
# }

# response = requests.post(url, json=payload)
# print(response.json())