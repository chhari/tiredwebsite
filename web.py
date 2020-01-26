from flask import Flask,render_template, request, send_file
import cv2
import numpy
#import infer_website as webinfer

app = Flask(__name__)

myImage = None
second=None
firstfilename=None
secondfilename=None


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success-table', methods=['POST'])
def success_table():
    global filename
    if request.method=="POST":
        file=request.files['file']
        try:
            firstfilename = file.filename
            filestr = request.files['file'].read()
            #convert string data to numpy array
            npimg = numpy.fromstring(filestr, numpy.uint8)
            # convert numpy array to image
            myImage = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            print(file)
            print("done mama")
            print(myImage)
            return render_template("index.html", text="read the file")
        except Exception as e:
            return render_template("index.html", text=str(e))

@app.route('/secondimage', methods=['POST'])
def secondimage():
    global filename
    if request.method=="POST":
        file2=request.files['file']
        try:
            secondfilename = file2.filename
            filestr2 = request.files['file'].read()
            #convert string data to numpy array
            npimg2 = numpy.fromstring(filestr2, numpy.uint8)
            # convert numpy array to image
            second = cv2.imdecode(npimg2, cv2.IMREAD_COLOR)
            print(file)
            print("second file mama")
            print(myImage)
            print(second)
            return render_template("index.html", text="read the second file")
        except Exception as e:
            return render_template("index.html", text=str(e))


@app.route('/ironman')
def ironman():
    if myImage is not None:
        webinfer.infer_method(myImage,second,firstfilename,"ironman")
        script1 = ""
        script1 += '<div class="col-lg-6 ">'
        script1 += '<a class="portfolio-box" href="static/img/%s">'%(a)
        script1 += '<img class="img-fluid" src="static/img/%s" alt="">'%(a)
        script1 += '</a>'
        script1 += '</div>'
        script1 += '<p></p>'
        return render_template("ironman.html",script1=script1)
    else:
        script1 = "image not available"
        return render_template("ironman.html",script1=script1)

@app.route('/changeback')
def changeback():
    script1 = ""
    if myImage is not None:
        webinfer.infer_method(myImage,second,firstfilename,"back")
        script1 += '<div class="col-lg-6 ">'
        script1 += '<a class="portfolio-box" href="static/img/%s">'%(a)
        script1 += '<img class="img-fluid" src="static/img/%s" alt="">'%(a)
        script1 += '</a>'
        script1 += '</div>'
        script1 += '<p></p>'
        return render_template("changeback.html",script1=script1)
    else:
        script1 = "image not available"
        return render_template("ironman.html",script1=script1)

@app.route('/stylize')
def stylize():
    if myImage is not None:
        webinfer.infer_method(myImage,second,firstfilename,"ironman")
        script1 = ""
        script1 += '<div class="col-lg-6 ">'
        script1 += '<a class="portfolio-box" href="static/img/%s">'%(a)
        script1 += '<img class="img-fluid" src="static/img/%s" alt="">'%(a)
        script1 += '</a>'
        script1 += '</div>'
        script1 += '<p></p>'
        return render_template("stylize.html",script1=script1)
    else:
        script1 = "image not available"
        return render_template("ironman.html",script1=script1)



if __name__=="__main__":
    app.run(debug=True)
