from flask import Flask, render_template, request
from werkzeug import secure_filename
app = Flask(__name__,template_folder='.')

@app.route('/')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      return render_template("result.html",result = result)

@app.route('/upload',methods = ['POST', 'GET'])
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploaded_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return render_template('template.html', filename = f.filename)

if __name__ == '__main__':
   app.run(debug = True)