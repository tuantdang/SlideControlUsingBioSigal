# Slideshow Control Using BioSigal
![Alt text](images/1.jpg?raw=true "Title")

# Build model
Refer to file BuildModel.py

# Run model and demo
Refer to file LoadModel.py

# Control Slide with snipcode


```` 
#Written by Tuan Dang
#Email: dangthanhtuanit@gmail.com, or tuan.dang@uta.edu
#Discoverred COM32 Windows API from C#, then applied for Python

import win32com.client
import time
Application = win32com.client.Dispatch("PowerPoint.Application")
Presentation = Application.Presentations.Open("D:\\dataset\\slide.pptx")
print(Presentation.Name)
Presentation.SlideShowSettings.Run()
time.sleep(3)
Presentation.SlideShowWindow.View.Next()
time.sleep(3)
Presentation.SlideShowWindow.View.Next()
time.sleep(3)
Presentation.SlideShowWindow.View.Previous()
time.sleep(3)
Application.Quit()
```` 

![Alt text](images/2.jpg?raw=true "Title")
![Alt text](images/3.jpg?raw=true "Title")
![Alt text](images/4.jpg?raw=true "Title")
![Alt text](images/5.jpg?raw=true "Title")
