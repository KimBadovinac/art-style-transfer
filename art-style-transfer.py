#Selfie camera imports
import pygame.camera
import pygame.image
import sys

#Style transfer imports
import os, time
import numpy as np
import tensorflow as tf #This module takes time to import
import tensorflow_hub as hub

#Display imports
import PIL

print("Done importing.")

dir_path = os.path.dirname(os.path.realpath(__file__))

pygame.init()
small_font = pygame.font.Font('freesansbold.ttf', 16)
font = pygame.font.SysFont(None, 30)
pygame.camera.init()
cameras = pygame.camera.list_cameras()
webcam = pygame.camera.Camera(cameras[0])

def take_selfie():
    selfiefilename = ""
    webcam.start()
    # grab first frame
    img = webcam.get_image()
    WIDTH = img.get_width()
    HEIGHT = img.get_height()
    screen = pygame.display.set_mode( ( WIDTH, HEIGHT ) )
    pygame.display.set_caption("Change your selfie into art")

    image_taken = False
    run = True
    while run:
        # draw frame
        screen.blit(img, (0,0))
        if image_taken: #selfie already taken, display buttons to retake or save selfie
            #button to retake image
            btn_retake_img = pygame.draw.rect(screen, (150,150,150), [WIDTH/2-100-10, HEIGHT-50, 110, 30], 0, 2)
            pygame.draw.rect(screen, (0,0,0), [WIDTH/2-100-10, HEIGHT-50, 110, 30], 2, 2)
            text = small_font.render("Retake selfie", True, (0,0,0))
            screen.blit(text, (WIDTH/2-100-10+5, HEIGHT-50+5+2))
            #button to save image
            btn_save_img = pygame.draw.rect(screen, (150,150,150), [WIDTH/2+10, HEIGHT-50, 100, 30], 0, 2)
            pygame.draw.rect(screen, (0,0,0), [WIDTH/2+10, HEIGHT-50, 100, 30], 2, 2)
            text = small_font.render("Save selfie", True, (0,0,0))
            screen.blit(text, (WIDTH/2+10+5, HEIGHT-50+5+2))
            #mouseover change color
            if btn_retake_img.collidepoint(pygame.mouse.get_pos()):
                btn_retake_img = pygame.draw.rect(screen, (100, 200, 255), [WIDTH/2-100-10, HEIGHT-50, 110, 30], 0, 2)
                pygame.draw.rect(screen, (0,0,0), [WIDTH/2-100-10, HEIGHT-50, 110, 30], 2, 2)
                text = small_font.render("Retake selfie", True, (0,0,0))
                screen.blit(text, (WIDTH/2-100-10+5, HEIGHT-50+5+2))
            if btn_save_img.collidepoint(pygame.mouse.get_pos()):
                btn_save_img = pygame.draw.rect(screen, (100, 200, 255), [WIDTH/2+10, HEIGHT-50, 100, 30], 0, 2)
                pygame.draw.rect(screen, (0,0,0), [WIDTH/2+10, HEIGHT-50, 100, 30], 2, 2)
                text = small_font.render("Save selfie", True, (0,0,0))
                screen.blit(text, (WIDTH/2+10+5, HEIGHT-50+5+2))
            for e in pygame.event.get():
                if e.type == pygame.MOUSEBUTTONDOWN:
                    if btn_retake_img.collidepoint(e.pos): #retake selfie
                        image_taken = False
                    if btn_save_img.collidepoint(e.pos): #save image
                        timestamp = time.strftime('%b-%d-%Y_%H%M%S', time.localtime())
                        selfiefilename = ("selfie-" + timestamp + ".jpg")
                        pygame.image.save(img, selfiefilename)
                        run = False
                        break
                if e.type == pygame.QUIT :
                    sys.exit()
        else: #selfie not yet taken
            #button to take image
            btn_take_img = pygame.draw.rect(screen, (150,150,150), [WIDTH/2-50, HEIGHT-50, 110, 30], 0, 2)
            pygame.draw.rect(screen, (0,0,0), [WIDTH/2-50, HEIGHT-50, 110, 30], 2, 2)
            text = small_font.render("Take selfie", True, (0,0,0))
            screen.blit(text, (WIDTH/2-50+5+5, HEIGHT-50+5+2))
            if btn_take_img.collidepoint(pygame.mouse.get_pos()):
                btn_take_img = pygame.draw.rect(screen, (100, 200, 255), [WIDTH/2-50, HEIGHT-50, 110, 30], 0, 2)
                pygame.draw.rect(screen, (0,0,0), [WIDTH/2-50, HEIGHT-50, 110, 30], 2, 2)
                text = small_font.render("Take selfie", True, (0,0,0))
                screen.blit(text, (WIDTH/2-50+5+5, HEIGHT-50+5+2))
            for e in pygame.event.get() :
                if e.type == pygame.QUIT :
                    sys.exit()
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE: #take image when SPACE is pressed
                        image_taken = True
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    if btn_take_img.collidepoint(e.pos): #take selfie
                        image_taken = True
            # grab next frame    
            img = webcam.get_image()
        
        pygame.display.flip()
    webcam.stop()
    return selfiefilename

# Style transfer part
#From: 
#https://paperswithcode.com/paper/exploring-the-structure-of-a-real-time#code

print("TF Version: ", tf.__version__)
print("TF Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

# Load TF Hub module.

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2' #Load from TensorFlow Hub
#hub_handle = dir_path + '/magenta_arbitrary-image-stylization-v1-256_2' #Load locally for faster start time
hub_module = hub.load(hub_handle)

style_urls = dict(
  tisnikar='file:///' + dir_path + '/tisnikar.jpg',
  kobilca='file:///' + dir_path + '/kobilca.jpg',
  jakopic='file:///' + dir_path + '/jakopic.jpg',
  malevic='file:///' + dir_path + '/malevic.jpg',
  warhol='file:///' + dir_path + '/warhol.jpg',
  warhol2='file:///' + dir_path + '/warhol3.jpg', #to show that you can change to different images of same art style
  warhol3='file:///' + dir_path + '/warhol4.jpg',
  warhol4='file:///' + dir_path + '/warhol5.jpg',
  warhol5='file:///' + dir_path + '/warhol6.jpg',
  stupica='file:///' + dir_path + '/stupica.jpg',
  cosic='file:///' + dir_path + '/cosic.jpg',
  ulay='file:///' + dir_path + '/ulay.jpg'
)

content_image_size = 384
style_image_size = 256
style_images = {k: load_image(v, (style_image_size, style_image_size)) for k, v in style_urls.items()}
style_images = {k: tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME') for k, style_image in style_images.items()}

content_name = 'selfie'
style_name = 'stupica'

#Display with pygame
class OptionBox():

    def __init__(self, x, y, w, h, color, highlight_color, font, option_list, selected = 5):
        self.color = color
        self.highlight_color = highlight_color
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.option_list = option_list
        self.selected = selected
        self.draw_menu = False
        self.menu_active = False
        self.active_option = -1

    def draw(self, surf):
        pygame.draw.rect(surf, self.highlight_color if self.menu_active else self.color, self.rect)
        pygame.draw.rect(surf, (0, 0, 0), self.rect, 2)
        msg = self.font.render(self.option_list[self.selected], 1, (0, 0, 0))
        surf.blit(msg, msg.get_rect(center = self.rect.center))

        if self.draw_menu:
            for i, text in enumerate(self.option_list):
                rect = self.rect.copy()
                rect.y += (i+1) * self.rect.height
                pygame.draw.rect(surf, self.highlight_color if i == self.active_option else self.color, rect)
                msg = self.font.render(text, 1, (0, 0, 0))
                surf.blit(msg, msg.get_rect(center = rect.center))
            outer_rect = (self.rect.x, self.rect.y + self.rect.height, self.rect.width, self.rect.height * len(self.option_list))
            pygame.draw.rect(surf, (0, 0, 0), outer_rect, 2)

    def update(self, event_list):
        mpos = pygame.mouse.get_pos()
        self.menu_active = self.rect.collidepoint(mpos)
        
        self.active_option = -1
        for i in range(len(self.option_list)):
            rect = self.rect.copy()
            rect.y += (i+1) * self.rect.height
            if rect.collidepoint(mpos):
                self.active_option = i
                break

        if not self.menu_active and self.active_option == -1:
            self.draw_menu = False

        for event in event_list:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.menu_active:
                    self.draw_menu = not self.draw_menu
                elif self.draw_menu and self.active_option >= 0:
                    self.selected = self.active_option
                    self.draw_menu = False
                    return self.active_option
        return -1

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

style_options = OptionBox(
    470, 80, 160, 40, (150, 150, 150), (100, 200, 255), pygame.font.SysFont(None, 30), 
    ["tisnikar", "kobilca", "jakopic", "malevic", "warhol", "stupica", "cosic", "ulay"])


run = True
selfie_mode = True
previous_selected_option = 5
warhol_selected_option = 0
while run: # loop listening for end of game
    if selfie_mode:
        selfiefilename = take_selfie()
        selfie_mode = False

        #Load selfie image
        content_urls = dict(
        #brad_pitt='https://images.mubicdn.net/images/cast_member/2552/cache-207-1524922850/image-w856.jpg',
        selfie='file:///' + dir_path + '/'+selfiefilename
        )
        content_images = {k: load_image(v, (content_image_size, content_image_size)) for k, v in content_urls.items()}

        # Stylize content image with given style image.
        # This is pretty fast within a few milliseconds on a GPU.

        stylized_image = hub_module(tf.constant(content_images[content_name]),
                                    tf.constant(style_images[style_name]))[0]

    event_list = pygame.event.get()

    selected_option = style_options.update(event_list)
    if previous_selected_option != selected_option:
        previous_selected_option = selected_option
        if selected_option == 0 and style_name != "tisnikar":
            style_name = "tisnikar"
            stylized_image = hub_module(tf.constant(content_images[content_name]),
                                tf.constant(style_images[style_name]))[0]
        elif selected_option == 1 and style_name != "kobilca":
            style_name = "kobilca"
            stylized_image = hub_module(tf.constant(content_images[content_name]),
                                tf.constant(style_images[style_name]))[0]
        elif selected_option == 2 and style_name != "jakopic":
            style_name = "jakopic"
            stylized_image = hub_module(tf.constant(content_images[content_name]),
                                tf.constant(style_images[style_name]))[0]
        elif selected_option == 3 and style_name != "malevic":
            style_name = "malevic"
            stylized_image = hub_module(tf.constant(content_images[content_name]),
                                tf.constant(style_images[style_name]))[0]
        elif selected_option == 4 and (style_name[0:6] != "warhol"):
            if warhol_selected_option == 0:
                style_name = "warhol"
            if warhol_selected_option == 1:
                style_name = "warhol2"
            if warhol_selected_option == 2:
                style_name = "warhol3"
            if warhol_selected_option == 3:
                style_name = "warhol4"
            if warhol_selected_option == 4:
                style_name = "warhol5"
            stylized_image = hub_module(tf.constant(content_images[content_name]),
                                tf.constant(style_images[style_name]))[0]
        elif selected_option == 5 and style_name != "stupica":
            style_name = "stupica"
            stylized_image = hub_module(tf.constant(content_images[content_name]),
                                tf.constant(style_images[style_name]))[0]
        elif selected_option == 6 and style_name != "cosic":
            style_name = "cosic"
            stylized_image = hub_module(tf.constant(content_images[content_name]),
                                tf.constant(style_images[style_name]))[0]
        elif selected_option == 7 and style_name != "ulay":
            style_name = "ulay"
            stylized_image = hub_module(tf.constant(content_images[content_name]),
                                tf.constant(style_images[style_name]))[0]
    
    pilImage = tensor_to_image(content_images[content_name])
    img1 = pygame.image.fromstring(pilImage.tobytes(), pilImage.size, pilImage.mode).convert()
    pilImage = tensor_to_image(style_images[style_name])
    img2 = pygame.image.fromstring(pilImage.tobytes(), pilImage.size, pilImage.mode).convert()
    pilImage = tensor_to_image(stylized_image)
    img3 = pygame.image.fromstring(pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

    WIDTH = 1100
    HEIGHT = 460
    screen = pygame.display.set_mode( ( WIDTH, HEIGHT ) )
    screen.fill((255, 255, 255))
    screen.blit(img1, (20,30)) # paint to screen
    screen.blit(img2, (420,158)) # paint to screen
    screen.blit(img3, (690,30)) # paint to screen
    style_options.draw(screen)
    #draw button for retaking selfie
    btn_retake_image = pygame.draw.rect(screen, (150,150,150), [470, 30, 160, 40])
    pygame.draw.rect(screen, (0,0,0), [470, 30, 160, 40], 2)
    text = font.render("Retake selfie", 1, (0, 0, 0))
    screen.blit(text, (470+15, 30+12))

    if btn_retake_image.collidepoint(pygame.mouse.get_pos()):
        btn_retake_image = pygame.draw.rect(screen, (100, 200, 255), [470, 30, 160, 40])
        pygame.draw.rect(screen, (0,0,0), [470, 30, 160, 40], 2)
        text = font.render("Retake selfie", 1, (0, 0, 0))
        screen.blit(text, (470+15, 30+12))

    for e in event_list:
        if e.type == pygame.MOUSEBUTTONDOWN:
            if btn_retake_image.collidepoint(e.pos): #retake selfie
                selfie_mode = True
        if e.type == pygame.KEYDOWN:
            #press any key to change to another image of same art style
            warhol_selected_option = (warhol_selected_option + 1) % 5
            if warhol_selected_option == 0:
                style_name = "warhol"
            if warhol_selected_option == 1:
                style_name = "warhol2"
            if warhol_selected_option == 2:
                style_name = "warhol3"
            if warhol_selected_option == 3:
                style_name = "warhol4"
            if warhol_selected_option == 4:
                style_name = "warhol5"
            stylized_image = hub_module(tf.constant(content_images[content_name]),
                            tf.constant(style_images[style_name]))[0]     

        if e.type == pygame.QUIT :
            sys.exit()

    pygame.display.flip() # paint screen one time

#loop over, quit pygame
pygame.quit()