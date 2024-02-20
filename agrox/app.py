from flask import Flask, render_template, request
import os
import cv2
from PIL import Image
from sklearn.cluster import KMeans
app = Flask(__name__)
def auto_detect_colors(image_path, num_colors=3):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels
    pixels = image_rgb.reshape((-1, 3))

    # Use k-means clustering to group similar colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    # Get the dominant colors
    dominant_colors = kmeans.cluster_centers_.astype(int)
    print(dominant_colors)
    return dominant_colors[2]
def count(p,x,img):
    flag = False
    cnt = 0
    for j in range(x,img.shape[1]):
        if (img[p, j][0]== 255 and img[p,j][1]==0 and img[p,j][2]==0 and flag == False):
            cnt = 0
            flag = True
            continue
        elif (img[p, j][0]== 255 and img[p,j][1]==0 and img[p,j][2]==0 and flag == True):
            break
        else:
            img[p, j] = (255, 255, 255)
            cnt+=1
    return cnt
def R(image_path):
    # Read the image using OpenCV
    img = Image.open(image_path)
    dpi = img.info.get('dpi')
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_img, 50,150)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to consider only larger rectangles
    min_contour_area = 1000
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    # Draw contours on the original image
    cv2.drawContours(img, filtered_contours, -1, (255, 0, 0), 1)
    object_height_cm = []
    M=0
    for i, contour in enumerate(filtered_contours):
        x, y, w, object_height_px = (cv2.boundingRect(contour))
        # cv2.rectangle(img, (x, y), (x + w, y + object_height_px), (255,255,255), 2)
        if (object_height_px > M):
            M = object_height_px
            x = cv2.boundingRect(contour)[0]
            y = cv2.boundingRect(contour)[1]
        object_height_cm.append(object_height_px / max(dpi) * 2.54)
    CF = 18.7 / min(object_height_cm)
    p2 = y + M // 4
    p1 = y + (3 * M) // 4
    p3 = y + M // 2
    H = round(CF  * max(object_height_cm),2)
    g75 = round(CF * count(p1, x, img) / max(dpi) * 2.54, 2)
    g50 = round(CF * count(p3, x, img) / max(dpi) * 2.54, 2)
    g25 = round(CF * count(p2, x, img) / max(dpi) * 2.54, 2)
    print("girth at 25%", CF*g25)
    print("girth at 50%", CF*g50)
    print("girth at 75%", CF*g75)
    print("ridge gourd height",H)
    return H,g25,g50,g75

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/action.html')
def action():
    return render_template('action.html')

@app.route('/upload1', methods=['POST'])
def upload1():
    file = request.files['file']
    if 'file' not in request.files:
        return render_template('action.html', result="No file part")
    if file.filename == '':
        return render_template('action.html', result="No selected file")
    if 'file' not in request.files:
        return render_template('action.html', result="No file part")
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    CN = str(auto_detect_colors(file_path))
    H,g25,g50,g75 = R(file_path)
    return render_template('action.html', result = "completed" ,height=H,g25 = g25,g50 = g50,g75 = g75,color_name = CN)


if __name__ == '__main__':
    app.run(debug=True)


