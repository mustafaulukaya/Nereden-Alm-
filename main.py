import collections
import imghdr
import tkinter
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk
from PIL import Image
from keras.models import model_from_json
import keras
import os
import os.path
from keras.optimizers import *
from keras.layers import *
from keras.models import *
from keras.applications.vgg16 import preprocess_input
from keras.regularizers import *
import mnist_reader as mr
import getpass
import cv2
import numpy as np
from colordescriptor import ColorDescriptor
from searcher import Searcher


img_height, img_width, num_classes = 200, 200, 4


def crop(canvas):
    canvas.data.colourPopToHappen = False
    canvas.data.drawOn = False
    # have to check if crop button is pressed or not, otherwise,
    # the root events which point to
    # different functions based on what button has been pressed
    # will get mixed up
    canvas.data.cropPopToHappen = True
    tkinter.messagebox.showinfo(title="Crop", message="Draw cropping rectangle and press Enter",
                                parent=canvas.data.mainWindow)
    if canvas.data.image is not None:
        canvas.data.mainWindow.bind("<ButtonPress-1>", lambda event: startcrop(event, canvas))
        canvas.data.mainWindow.bind("<B1-Motion>", lambda event: drawcrop(event, canvas))
        canvas.data.mainWindow.bind("<ButtonRelease-1>", lambda event: endcrop(event, canvas))


def startcrop(event, canvas):
    # detects the start of the crop rectangle
    if canvas.data.endcrop is False and canvas.data.cropPopToHappen:
        canvas.data.startCropX = event.x
        canvas.data.startCropY = event.y


def drawcrop(event, canvas):
    # keeps extending the crop rectange as the user extends
    # his desired crop rectangle
    if canvas.data.endcrop is False and canvas.data.cropPopToHappen:
        canvas.data.tempCropX = event.x
        canvas.data.tempCropY = event.y
        canvas.create_rectangle(canvas.data.startCropX, canvas.data.startCropY, canvas.data.tempCropX,
                                canvas.data.tempCropY, fill="gray", stipple="gray12", width=0)


def endcrop(event, canvas):
    # set canvas.data.endcrop=True so that button pressed movements
    # are not caught anymore but set it to False when "Enter"
    # is pressed so that crop can be performed another time too
    if canvas.data.cropPopToHappen:
        canvas.data.endcrop = True
        canvas.data.endCropX = event.x
        canvas.data.endCropY = event.y
        canvas.create_rectangle(canvas.data.startCropX, canvas.data.startCropY, canvas.data.endCropX,
                                canvas.data.endCropY, fill="gray", stipple="gray12", width=1)
        canvas.data.mainWindow.bind("<Return>", lambda event: performcrop(event, canvas))


def performcrop(event, canvas):
    canvas.data.image = canvas.data.image.crop((int(
        round((canvas.data.startCropX - canvas.data.imageTopX) * canvas.data.imageScale)), int(
        round((canvas.data.startCropY - canvas.data.imageTopY) * canvas.data.imageScale)), int(
        round((canvas.data.endCropX - canvas.data.imageTopX) * canvas.data.imageScale)), int(
        round((canvas.data.endCropY - canvas.data.imageTopY) * canvas.data.imageScale))))
    canvas.data.endcrop = False
    canvas.data.cropPopToHappen = False
    canvas.data.undoQueue.append(canvas.data.image.copy())
    canvas.data.imageForTk = makeimagefortk(canvas)
    drawimage(canvas)


def save(canvas):
    if canvas.data.image is not None:
        im = canvas.data.image
        im.save(canvas.data.imageLocation)


def saveas(canvas):
    # ask where the user wants to save the file
    if canvas.data.image is not None:
        filename = tkinter.filedialog.asksaveasfilename(defaultextension=".jpg")
        im = canvas.data.image
        im.save(filename)


def drawimage(canvas):
    if canvas.data.image is not None:
        # make the canvas center and the image center the same
        canvas.create_image(canvas.data.width / 2.0 - canvas.data.resizedIm.size[0] / 2.0,
                            canvas.data.height / 2.0 - canvas.data.resizedIm.size[1] / 2.0, anchor=tkinter.NW,
                            image=canvas.data.imageForTk)
        canvas.data.imageTopX = int(round(canvas.data.width / 2.0 - canvas.data.resizedIm.size[0] / 2.0))
        canvas.data.imageTopY = int(round(canvas.data.height / 2.0 - canvas.data.resizedIm.size[1] / 2.0))


def makeimagefortk(canvas):
    im = canvas.data.image
    if canvas.data.image is not None:
        # Beacuse after cropping the now 'image' might have diffrent
        # dimensional ratios
        imagewidth = canvas.data.image.size[0]
        imageheight = canvas.data.image.size[1]
        # To make biggest version of the image fit inside the canvas
        if imagewidth > imageheight:
            resizedimage = im.resize(
                (canvas.data.width, int(round(float(imageheight) * canvas.data.width / imagewidth))))
            # store the scale so as to use it later
            canvas.data.imageScale = float(imagewidth) / canvas.data.width
        else:
            resizedimage = im.resize(
                (int(round(float(imagewidth) * canvas.data.height / imageheight)), canvas.data.height))
            canvas.data.imageScale = float(imageheight) / canvas.data.height
        # we may need to refer to ther resized image atttributes again
        canvas.data.resizedIm = resizedimage
        return ImageTk.PhotoImage(resizedimage)


def newimage(canvas):
    imagename = tkinter.filedialog.askopenfilename()
    # make sure it's an image file
    filetype = imghdr.what(imagename)
    if filetype in ['jpeg', 'bmp', 'png', 'tiff']:
        canvas.data.imageLocation = imagename
        im = Image.open(imagename)
        canvas.data.image = im
        canvas.data.originalImage = im.copy()
        canvas.data.undoQueue.append(im.copy())
        canvas.data.imageSize = im.size  # Original Image dimensions
        canvas.data.imageForTk = makeimagefortk(canvas)
        drawimage(canvas)
    elif filetype not in ['jpeg', 'bmp', 'png', 'tiff']:
        tkinter.messagebox.showinfo(title="Image File", message="Choose an Image File!", parent=canvas.data.mainWindow)
    # restrict filetypes to .jpg, .bmp, etc.


def undo(canvas):
    if len(canvas.data.undoQueue) > 0:
        # the last element of the Undo Deque is the
        # current version of the image
        lastimage = canvas.data.undoQueue.pop()
        # we would want the current version if wehit redo after undo
        canvas.data.redoQueue.appendleft(lastimage)
    if len(canvas.data.undoQueue) > 0:
        # the previous version of the image
        canvas.data.image = canvas.data.undoQueue[-1]
    canvas.data.imageForTk = makeimagefortk(canvas)
    drawimage(canvas)


def redo(canvas):
    if len(canvas.data.redoQueue) > 0:
        canvas.data.image = canvas.data.redoQueue[0]
    if len(canvas.data.redoQueue) > 0:
        # we remove this version from the Redo Deque beacuase it
        # has become our current image
        lastimage = canvas.data.redoQueue.popleft()
        canvas.data.undoQueue.append(lastimage)
    canvas.data.imageForTk = makeimagefortk(canvas)
    drawimage(canvas)


def keypressed(canvas, event):
    if event.keysym == "z":
        undo(canvas)
    elif event.keysym == "y":
        redo(canvas)


def init(root, canvas):
    buttonsinit(root, canvas)
    canvas.data.image = None
    canvas.data.angleSelected = None
    canvas.data.colourPopToHappen = False
    canvas.data.cropPopToHappen = False
    canvas.data.endcrop = False
    canvas.data.drawOn = True

    canvas.data.undoQueue = collections.deque([], 10)
    canvas.data.redoQueue = collections.deque([], 10)
    canvas.pack()


def buttonsinit(root, canvas):
    backgroundcolour = "white"
    buttonwidth = 7
    buttonheight = 1
    toolkitframe = tkinter.Frame(root)

    cropbutton = tkinter.Button(toolkitframe, text="New", background=backgroundcolour, width=buttonwidth,
                                height=buttonheight, command=lambda: newimage(canvas))
    cropbutton.grid(row=0, column=0)

    cropbutton = tkinter.Button(toolkitframe, text="Crop", background=backgroundcolour, width=buttonwidth,
                                height=buttonheight, command=lambda: crop(canvas))
    cropbutton.grid(row=0, column=1)
    cropbutton = tkinter.Button(toolkitframe, text="Undo", background=backgroundcolour, width=buttonwidth,
                                height=buttonheight, command=lambda: undo(canvas))
    cropbutton.grid(row=0, column=2)
    cropbutton = tkinter.Button(toolkitframe, text="Redo", background=backgroundcolour, width=buttonwidth,
                                height=buttonheight, command=lambda: redo(canvas))
    cropbutton.grid(row=0, column=3)
    cropbutton = tkinter.Button(toolkitframe, text="Query", background=backgroundcolour, width=buttonwidth,
                                height=buttonheight, command=lambda: query(canvas))
    cropbutton.grid(row=0, column=4)

    cropbutton = tkinter.Button(toolkitframe, text="Test", background=backgroundcolour, width=buttonwidth,
                                height=buttonheight, command=lambda: test())
    cropbutton.grid(row=0, column=5)

    cropbutton = tkinter.Button(toolkitframe, text="CBIR", background=backgroundcolour, width=buttonwidth,
                                height=buttonheight, command=lambda: getSimilarImage(canvas))
    cropbutton.grid(row=0, column=6)

    toolkitframe.pack(side=tkinter.TOP)


def test():

    x_test, y_test = mr.load_mnist('dataset/dataset', kind='t10k')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_testtest = np.ndarray(shape=(10000, img_height, img_width, 3), dtype=int)

    for i in range(0, 10000):
        tempimg = x_test[i]
        tempimg = cv2.cvtColor(tempimg, cv2.COLOR_GRAY2RGB)
        tempimg = cv2.resize(tempimg, (img_width, img_height))
        x_testtest[i] = tempimg

    x_test = x_testtest

    y_test = keras.utils.to_categorical(y_test, num_classes)
    score = model.evaluate(x_test, y_test, verbose=1)

    print('')
    print(np.ravel(model.predict(x_test)))
    print('training data results: ')
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(score[i]))


def run():
    # create the root and the canvas
    root = tkinter.Tk()
    root.title("Content Base Image Retrieval")
    canvaswidth = 1280
    canvasheight = 724
    canvas = tkinter.Canvas(root, width=canvaswidth, height=canvasheight, background="gray")

    # Set up canvas data and call init

    class Struct: pass

    canvas.data = Struct()
    canvas.data.width = canvaswidth
    canvas.data.height = canvasheight
    canvas.data.mainWindow = root
    init(root, canvas)
    root.bind("<Key>", lambda event: keypressed(canvas, event))
    # and launch the app
    root.mainloop()  # This call BLOCKS (so your program waits)


def loadmodel():
    # username = getpass.getuser()
    # print(os.path.dirname(os.path.abspath(__file__)))

    PATH1 = 'model.json'
    PATH1 = os.path.join(os.getcwd() + '/saves', PATH1)

    PATH2 = 'model.h5'
    PATH2 = os.path.join(os.getcwd() + '/saves', PATH2)

    if not (os.path.isfile(os.getcwd() + "/index/0.csv") and os.access(os.getcwd() + "/index/0.csv", os.R_OK)):
        print("T-shirt indexleniyor")
        os.system("python index.py --dataset T-shirt --index 0.csv")

    if not (os.path.isfile(os.getcwd() + "/index/1.csv") and os.access(os.getcwd() + "/index/1.csv", os.R_OK)):
        print("Pantalon indexleniyor")
        os.system("python index.py --dataset Pantalon --index 1.csv")

    if not (os.path.isfile(os.getcwd() + "/index/2.csv") and os.access(os.getcwd() + "/index/2.csv", os.R_OK)):
        print("Suveter indexleniyor")
        os.system("python index.py --dataset Suveter --index 2.csv")

    if not (os.path.isfile(os.getcwd() + "/index/3.csv") and os.access(os.getcwd() + "/index/3.csv", os.R_OK)):
        print("Elbise indexleniyor")
        os.system("python index.py --dataset Elbise --index 3.csv")

    if not (os.path.isfile(os.getcwd() + "/index/4.csv") and os.access(os.getcwd() + "/index/4.csv", os.R_OK)):
        print("Kaban indexleniyor")
        os.system("python index.py --dataset Kaban --index 4.csv")

    if not (os.path.isfile(os.getcwd() + "/index/5.csv") and os.access(os.getcwd() + "/index/5.csv", os.R_OK)):
        print("Sandalet-Terlik indexleniyor")
        os.system("python index.py --dataset Sandalet-Terlik --index 5.csv")

    if not (os.path.isfile(os.getcwd() + "/index/6.csv") and os.access(os.getcwd() + "/index/6.csv", os.R_OK)):
        print("Gomlek indexleniyor")
        os.system("python index.py --dataset Gomlek --index 6.csv")

    if not (os.path.isfile(os.getcwd() + "/index/7.csv") and os.access(os.getcwd() + "/index/7.csv", os.R_OK)):
        print("Ayakkabi indexleniyor")
        os.system("python index.py --dataset Ayakkabi --index 7.csv")

    if not (os.path.isfile(os.getcwd() + "/index/8.csv") and os.access(os.getcwd() + "/index/8.csv", os.R_OK)):
        print("Canta indexleniyor")
        os.system("python index.py --dataset Canta --index 8.csv")

    if not (os.path.isfile(os.getcwd() + "/index/9.csv") and os.access(os.getcwd() + "/index/9.csv", os.R_OK)):
        print("Cizme indexleniyor")
        os.system("python index.py --dataset Cizme --index 9.csv")

    print("Indexleme bitti. Model kontrolleri başlıyor.")

    if os.path.isfile(PATH1) and os.access(PATH1, os.R_OK) and os.path.isfile(PATH2) and os.access(PATH2, os.R_OK):
        json_file = open(PATH1, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(PATH2)
        print("Loaded model from disk")

        # Optimizasyon yöntemi ve hiperparametreler
        # myopt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, momentum=0.9, amsgrad=False, nesterov=True)

        # Modeli derledik
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        return model

    else:
        # Sinir Ağımızı Oluşturuyoruz

        batch_size = 128
        epochs = 100
        # num_classes = 10

        # img_width, img_height = 28, 28  # ağa girecek fotoğrafların boyutu

        # 3-channel kabul ediyor,gray scale sokamıyoruz ve en düşük 48x48 lik fotoğraf istiyor
        # pre train için ağlar hakkında bilgi: https://keras.io/applications

        '''
        base_model = keras.applications.VGG16(include_top=False,
                                              weights='imagenet',
                                              input_shape=(img_width, img_height, 3),
                                              pooling='avg')
        '''
        '''
        base_model = keras.applications.InceptionResNetV2(include_top=False,
                                                          weights='imagenet',
                                                          input_shape=(img_width, img_height, 3),
                                                          pooling='avg') #depth = 572


        '''
        '''
        x = base_model.output
        x = (Flatten())(x)
        x = (Dense(512))(x)
        x = (Activation('relu'))(x)
        x = (Dropout(0.5))(x)
        x = (Dense(num_classes, activation='softmax'))(x)
        '''
        '''
        inputs = Input((img_width, img_height, 3))

        x = Lambda(preprocess_input, name='preprocessing')(inputs)
        outputs = base_model(x)
        outputs = Dropout(0.5)(outputs)
        outputs = Dense(num_classes,activation='softmax')(outputs)
        model = Model(inputs, outputs)
        '''

        model = Sequential()
        model.add(
            Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01), input_shape=(img_width, img_height, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5, 5), kernel_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (5, 5), kernel_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # Modeli derleyip hazırladık, geriye eğitmek kaldı
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        # Kendi veri setimize geçtik

        # dosya isimleri class'lar, içeriği de train olacak şekilde ayarlamam gerek


        # fashion_mnist datasetleri x'ler image y'ler label
        x_train, y_train = mr.load_mnist('dataset/dataset', kind='train')
        x_test, y_test = mr.load_mnist('dataset/dataset', kind='t10k')

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)




        # RGB için extra axis ekledik
        x_train = np.expand_dims(x_train, axis=0)
        x_test = np.expand_dims(x_test, axis=0)


        x_traintest = np.ndarray(shape=(60000, img_height, img_width, 3), dtype=int)
        
        # gray'dan RGB ye çevirdik
        for i in range(0, 60000):
            tempimg = x_train[i]
            tempimg = cv2.cvtColor(tempimg, cv2.COLOR_GRAY2RGB)
            tempimg = cv2.resize(tempimg, (img_width, img_height))
            x_traintest[i] = tempimg

        x_train = x_traintest

        x_testtest = np.ndarray(shape=(10000, img_height, img_width, 3), dtype=int)

        for i in range(0, 10000):
            tempimg = x_test[i]
            tempimg = cv2.cvtColor(tempimg, cv2.COLOR_GRAY2RGB)
            tempimg = cv2.resize(tempimg, (img_width, img_height))
            x_testtest[i] = tempimg

        x_test = x_testtest

        # Veri setimizdeki etiketleri de 1 boyutlu numpy array yapısından kategorik matriks yapısına çevirmemiz gerekiyor.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)


        # Optimizasyon yöntemi ve hiperparametreler
        # myopt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        # eğitim
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  verbose=1,
                  shuffle=True)

        score = model.evaluate(x_test, y_test, verbose=1)

        print('')
        print(np.ravel(model.predict(x_test)))
        print('training data results: ')
        for i in range(len(model.metrics_names)):
            print(str(model.metrics_names[i]) + ": " + str(score[i]))

        # modeli eğittikten sonra kaydediyoruz
        model_json = model.to_json()
        with open(PATH1, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(PATH2)
        print("Saved model to disk")

        return model


def query(canvas):
    loc = canvas
    loc.data.imageLocation = loc.data.imageLocation.strip('.jpg') + '_cropped.jpg'
    save(loc)
    im = cv2.imread(loc.data.imageLocation, 1)
    im = cv2.resize(im, (28, 28))

    x = np.ndarray(shape=(1, 28, 28, 3), dtype=int)
    x[0] = im

    predict = model.predict_classes(x)

    if predict == 0:
        print('T-shirt')

    elif predict == 1:
        print('Pantalon')

    elif predict == 2:
        print('Suveter')

    elif predict == 3:
        print('Elbise')

    elif predict == 4:
        print('Kaban')

    elif predict == 5:
        print('Sandalet-Terlik')

    elif predict == 6:
        print('Gomlek')

    elif predict == 7:
        print('Ayakkabi')

    elif predict == 8:
        print('Canta')

    elif predict == 9:
        print('Cizme')


def concat_images(imga, imgb):
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img


def getSimilarImage(canvas):
    image = np.array(canvas.data.image)

    loc = canvas
    loc.data.imageLocation = loc.data.imageLocation.strip('.png') + '_cropped.png'
    save(loc)
    im = cv2.imread(loc.data.imageLocation, 1)
    os.remove(loc.data.imageLocation)

    im = cv2.resize(im, (28, 28))

    x = np.ndarray(shape=(1, 28, 28, 3), dtype=int)
    x[0] = im

    numofclass = model.predict_classes(x)

    index = os.getcwd() + "/index/" + str(numofclass) + ".csv"
    index = index.replace("[", "")
    index = index.replace("]", "")

    cd = ColorDescriptor((8, 12, 3))
    features = cd.describe(image)

    searcher = Searcher(index)
    results = searcher.search(features)

    for (score, resultID) in results:
        # load the result image and display it
        result = cv2.imread("dataset/dataset/images/" + resultID)
        cv2.imshow("Result", result)
        image = concat_images(image, result)

    # cv2. resize(image, (image.size[0]*2, image.size[1]*2))
    cv2.imshow("Result", image)


model = loadmodel()
model.summary()
run()
