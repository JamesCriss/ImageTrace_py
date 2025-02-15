#from imageTrace import convertimage
import webview

# main.py


# Call the convertimage function
#convertimage()

# Create a webview window
# Set the window icon and background color
webview.create_window('Image Trace', 'interface/index.html', width=1280, height=900, frameless=True)

# Start the webview event loop
webview.start()