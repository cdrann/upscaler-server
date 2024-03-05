import requests

url = 'http://localhost:8000'
files = {'image': open('C://Users//Admin//Desktop//upscaler_server//294599.jpg', 'rb')}

response = requests.post(url, files=files)
processed_image = response.content
# Обработка полученного изображения

# Сохранение обработанного изображения в файл
with open('processed_image.jpg', 'wb') as f:
    f.write(processed_image)

print('Processed image saved as processed_image.jpg')

