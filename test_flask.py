from flask import *
import flask_test_file as out
from werkzeug import secure_filename

app = Flask(__name__)

@app.route('/')
def hello_world():
  return """ 


  <html>
  <body>
  <form action="/send" method = "post" enctype = "multipart/form-data"> 
      <input type="file" id="myFile" name="myFile"/>
      <label for="fileupload"> Select a file to upload</label> 
      <br><br><input type="Submit">
      
  </form>
  </body>
  </html>



   """

@app.route('/send',methods = ['POST'])
def hello_world1():
  if request.method == 'POST':
    f = request.files['myFile']
    f.save("C:\\Users\\abdul\\Downloads\\object_detection_demo-master\\static\\test_images\\"+f.filename)  
    return "image_saved"
    #f.save(secure_filename(f.filename)
   
  # ans = out.output(file)
  # full_filename = "C:/Users/abdul/Downloads/object_detection_demo-master/static/img1.png"
  # return render_template("image.html",user_image = full_filename)
if __name__ == '__main__':
  app.run(debug = True)


# <form action="/send" method = "post"> 
#    <input type="file" name="fileupload" value="fileupload" id="fileupload"> 
#    <label for="fileupload"> Select a file to upload</label> 
#    <br><br><input type="Submit">
#    </form>