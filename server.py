from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
from PIL import Image
# from diffusers import StableDiffusionUpscalePipeline
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion_superresolution import LDMSuperResolutionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')
model_id = "CompVis/ldm-super-resolution-4x-openimages"
# model_id = "stabilityai/stable-diffusion-x4-upscaler"

# load model and scheduler
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
# pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to(device)


class DesktopUpscaler:
    def resize_image(self, image, max_size):
        # Получаем текущие размеры изображения
        original_width, original_height = image.size

        # Вычисляем соотношение сторон изображения
        aspect_ratio = original_width / original_height

        # Проверяем, превышает ли какая-либо из сторон максимальный размер
        if original_width > max_size or original_height > max_size:
            if aspect_ratio > 1:
                # Ширина больше высоты, уменьшаем по ширине
                new_width = max_size
                new_height = int(max_size / aspect_ratio)
            else:
                # Высота больше ширины, уменьшаем по высоте
                new_width = int(max_size * aspect_ratio)
                new_height = max_size
        
            # Изменяем размер изображения
            image = image.resize((new_width, new_height), Image.LANCZOS)
    
        return image

    def load_and_resize_image(self, image):
        # Загружаем изображение и преобразуем его в RGB
        #image = Image.open(file_path).convert("RGB")

        #   Максимальный размер для одной из сторон изображения
        max_size = 256
        # Вызываем функцию изменения размера
        image = self.resize_image(image, max_size)

        return image
 
    def upscale_image(self, image, pipeline=pipeline):
        low_res_img = self.load_and_resize_image(image)
        upscaled_image = pipeline(low_res_img, num_inference_steps=50, eta=1).images[0]
        return upscaled_image
    

class ImageHandler(BaseHTTPRequestHandler):
    
    def do_POST(self):
        content_type, params = cgi.parse_header(self.headers['content-type'])
        
        if content_type == 'multipart/form-data':
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            
            image_data = form['image'].file.read()
            
            # Обработка изображения
            desktop_upscaler = DesktopUpscaler()
            new_image = desktop_upscaler.upscale_image(image_data, pipeline)
            
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            self.wfile.write(new_image)
    

def run(server_class=HTTPServer, handler_class=ImageHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run()