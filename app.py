from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from markupsafe import escape
import cv2
import numpy as np


app = Flask(__name__)


UPLOAD_FOLDER = 'storage'
EDITED_FOLDER = 'storage\edited'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EDITED_FOLDER'] = EDITED_FOLDER
app.config['SECRET_KEY'] = 'kjsdcksdkhsduiuqwye9y92191781732198327918742kjdsnkljnw'

filenames = []
description_dict = {'sobel_vertical' : 'cv2.Sobel(image, ddepth=-1, ksize=3, dx=1, dy=0)',
                    'sobel_horizontal' : 'cv2.Sobel(image, ddepth=-1, ksize=3, dx=0, dy=1)',
                    'schar_vertical' : 'cv2.Scharr(image,ddepth=-1,dx=1,dy=0)',
                    'schar_horizontal' : 'cv2.Scharr(image,ddepth=-1,dx=0,dy=1)',
                    'laplacian' : 'cv2.Laplacian(image, ksize=3, ddepth=-1)',
                    'black_hat' : 'cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel=np.ones((5, 5), np.uint8))',
                    'top_hat' : 'cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel=np.ones((5, 5), np.uint8))',
                    'morphological_gradient' : 'cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel=np.ones((5, 5), np.uint8)) ',
                    'opening' : 'cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8), iterations=1)',
                    'closing' : 'cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8), iterations=1)',
                    'gabor' : 'cv2.filter2D(image, cv2.CV_8UC3, kernel=cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F))',
                    'gaussian_blur' : 'cv2.GaussianBlur(image, (7,7), 0)',
                    'erode' : 'cv2.erode(image, kernel=np.ones((5, 5), np.uint8), iterations=1)',
                    'dilate' : 'cv2.dilate(image, kernel=np.ones((5, 5), np.uint8), iterations=1)',
                    'box_filter' : 'cv2.boxFilter(image, ddepth=-20, ksize=(5, 5))',
                    'blur' : 'cv2.blur(image, ksize=(5, 5))',
                    'bilateral_filter' : 'cv2.bilateralFilter(image, d=20, sigmaColor=75, sigmaSpace=75)'}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filenames.append(filename)

            path = url_for('display_image', filename=filename)
            print(path)
            a = UPLOAD_FOLDER+'\\'+str(filenames[-1]) # path to uploaded file in a storage folder
            #print(a)
            return render_template('page_3.html', filename=filename, path=path)
    
    return render_template("index.html")
    
print(filenames)

@app.route('/UPLOAD_FOLDER/<filename>')
def display_image(filename):
    if filename:
    
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
    return render_template('page_2.html', filename=filename)

@app.route('/EDITED_FOLDER/<filename>')
def display_edited(filename):
    if filename:
    
        return send_from_directory(app.config['EDITED_FOLDER'] , filename)
    
    return render_template('page_2.html', filename=filename)

@app.route('/sobel_vertical', methods=['GET'])
def sobel_vertical():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    sobel_vertical_img = cv2.Sobel(newImage, ddepth=-1, ksize=3, dx=1, dy=0)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'sobel_vertical_img.jpg'), sobel_vertical_img) 
    filename = 'sobel_vertical_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['sobel_vertical']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/sobel_horizontal', methods=['GET'])
def sobel_horizontal():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    sobel_horizontal_img = cv2.Sobel(newImage,ddepth=-1,ksize=3,dx=0,dy=1)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'sobel_horizontal_img.jpg'), sobel_horizontal_img) 
    filename = 'sobel_horizontal_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['sobel_horizontal']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/schar_vertical', methods=['GET'])
def schar_vertical():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    schar_vertical_img = cv2.Scharr(newImage,ddepth=-1,dx=1,dy=0)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'schar_vertical_img.jpg'), schar_vertical_img) 
    filename = 'schar_vertical_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['schar_vertical']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/schar_horizontal', methods=['GET'])
def schar_horizontal():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    schar_horizontal_img = cv2.Scharr(newImage,ddepth=-1,dx=0,dy=1)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'schar_horizontal_img.jpg'), schar_horizontal_img) 
    filename = 'schar_horizontal_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['schar_horizontal']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/laplacian', methods=['GET'])
def laplacian():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    Laplacian_img = cv2.Laplacian(newImage, ksize=3, ddepth=-1)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'Laplacian_img.jpg'), Laplacian_img) 
    filename = 'Laplacian_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['laplacian']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/black_hat', methods=['GET'])
def blackhat():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    kernel = np.ones((5, 5), np.uint8)
    BlackHat_img = cv2.morphologyEx(newImage, cv2.MORPH_BLACKHAT, kernel)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'BlackHat_img.jpg'), BlackHat_img) 
    filename = 'BlackHat_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['black_hat']
    return render_template('page_3.html', path=path, filename=filename, description=description)


@app.route('/top_hat', methods=['GET'])
def tophat():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    kernel = np.ones((5, 5), np.uint8)
    TopHat_img = cv2.morphologyEx(newImage, cv2.MORPH_TOPHAT, kernel) 

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'TopHat_img.jpg'), TopHat_img) 
    filename = 'TopHat_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['top_hat']
    return render_template('page_3.html', path=path, filename=filename, description=description) 


@app.route('/morphological_gradient', methods=['GET'])
def morphological_gradient():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    kernel = np.ones((5, 5), np.uint8)
    Morphological_Gradient_img = cv2.morphologyEx(newImage, cv2.MORPH_GRADIENT, kernel) 

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'Morphological_Gradient_img.jpg'), Morphological_Gradient_img) 
    filename = 'Morphological_Gradient_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['morphological_gradient']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/opening', methods=['GET'])
def opening():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    kernel = np.ones((5, 5), np.uint8)
    Opening_img = cv2.morphologyEx(newImage, cv2.MORPH_OPEN, kernel, iterations=1)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'Opening_img.jpg'), Opening_img) 
    filename = 'Opening_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['opening']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/closing', methods=['GET'])
def closing():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    kernel = np.ones((5, 5), np.uint8)
    Closing_img = cv2.morphologyEx(newImage, cv2.MORPH_CLOSE, kernel, iterations=1)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'Closing_img.jpg'), Closing_img) 
    filename = 'Closing_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['closing']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/gabor', methods=['GET'])
def gabor():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    ksize = 31 # It is the size of the filter kernel (width, height) in pixels. This can be a tuple or a single integer value.
    sigma = 0.8 # Standard deviation of the Gaussian envelope
    theta = np.pi/2 # Orientation of the filter in degrees
    lamda = 3*np.pi/4 # Wavelength of the sinusoidal factor
    gamma = 2 # Spatial aspect ratio (ellipticity) of the filter
    phi = 0 # Phase offset of the filter in degrees. This is an optional parameter and its default value is 0

    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    gabor_img = cv2.filter2D(newImage, cv2.CV_8UC3, kernel)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'gabor_img.jpg'), gabor_img) 
    filename = 'gabor_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['gabor']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/gaussian_blur', methods=['GET'])
def gaussian_blur():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    gaussian_blur_img = cv2.GaussianBlur(newImage, (7,7), 0)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'gaussian_blur_img.jpg'), gaussian_blur_img) 
    filename = 'gaussian_blur_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['gaussian_blur']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/erode', methods=['GET'])
def erode():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    kernel = np.ones((5, 5), np.uint8)
    erode_img = cv2.erode(newImage, kernel, iterations=1)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'erode_img.jpg'), erode_img) 
    filename = 'erode_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['erode']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/dilate', methods=['GET'])
def dilate():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    kernel = np.ones((5, 5), np.uint8)
    dilate_img = cv2.dilate(newImage, kernel, iterations=1)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'dilate_img.jpg'), dilate_img) 
    filename = 'dilate_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['dilate']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/box_filter', methods=['GET'])
def box_filter():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    box_filter_img = cv2.boxFilter(newImage, ddepth=-20, ksize=(5, 5))

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'box_filter_img.jpg'), box_filter_img) 
    filename = 'box_filter_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['box_filter']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/blur', methods=['GET'])
def blur():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    blur_img = cv2.blur(newImage, ksize=(5, 5))

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'blur_img.jpg'), blur_img) 
    filename = 'blur_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['blur']
    return render_template('page_3.html', path=path, filename=filename, description=description)

@app.route('/bilateral_filter', methods=['GET'])
def bilateral_filter():
    original_image = UPLOAD_FOLDER+'\\'+str(filenames[-1])
    image = cv2.imread(original_image)
    newImage = image.copy()

    bilateral_filter_img = cv2.bilateralFilter(newImage, d=20, sigmaColor=75, sigmaSpace=75)

    cv2.imwrite(os.path.join(EDITED_FOLDER, 'bilateral_filter_img.jpg'), bilateral_filter_img) 
    filename = 'bilateral_filter_img.jpg'
    path = url_for('display_edited', filename=filename)
    description = description_dict['bilateral_filter']
    return render_template('page_3.html', path=path, filename=filename, description=description)

if __name__ == '__main__':
    app.run(debug=False)