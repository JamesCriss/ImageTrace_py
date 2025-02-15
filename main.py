#from imageTrace import convertimage
import webview
import os




def open_file_dialog():
        file_types = ('Image Files (*.png;*.jpg;*.jpeg)', 'All files (*.*)')
        file_path = webview.windows[0].create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types)
        if file_path:
            return file_path[0]
        return None

def load_image(js_api, image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    js_api.set_image(image_data)

class Api:
    def open_file(self):
        file_path = open_file_dialog()
        if file_path:
            load_image(self, file_path)

    def set_image(self, image_data):
        webview.windows[0].evaluate_js(f"setImage('{image_data}')")

    def button1_clicked(self):
        print("clicked")
        self.open_file()


if __name__ == '__main__':
    api = Api()
    webview.create_window('Image Trace', 'interface/index.html', width=1280, height=800, frameless=True, js_api=api)

    
    webview.start()

   

   