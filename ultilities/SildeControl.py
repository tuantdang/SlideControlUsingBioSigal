#Written by Tuan Dang
#Email: dangthanhtuanit@gmail.com, or tuan.dang@uta.edu
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

