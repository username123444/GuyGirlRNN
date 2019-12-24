from trainer import *
from crawler import *
import math
import sys

def show_help_menu():
    print("main.py:")
    print("\n\t--train [--clear-output is optional]: Trains the neural network based on the datasets in datasets/naked and datasets/clothed")
    print("\n\t--test-images [TEST_DIRECTORY] [--verify is optional]: Tests all of the images in the given directory. --verify allows for you to verify the category of every image")
    print("\n\t--crawl-google [GOOGLE_IMAGES_SEARCH_QUERY] [DOWNLOAD_DIRECTORY]: Downloads a maximum of 10 images from the given google images search to the download directory")
    print("\n\t--single-image-train [IMAGE_PATH] [IMAGE_CATEGORY] [--clear-output is optional]: Trains the network on a single image, requires that the network has been trained before")
try:
    __categories = ["naked", "clothed"]
    #Check if the user wants to train the network
    if(sys.argv[1] == "--train"):
        #Initialize a clear output variable
        __clear_output = False
        #Check if the developer wants to clear the output
        if("--clear-output" in sys.argv):
            #Set the clear output variable to true
            __clear_output = True
        #Initialize our trainer
        __trainer = Trainer("datasets/", __categories, clear_output=__clear_output)
        #Train the network with the given parameters
        __trainer.train_image_model(epochs=100, train_times=1, dense_layers=[0], layer_sizes=[64], convolusion_layers=[3], batch_size=Trainer.BATCH_SIZE_DEFAULT, validation_split=0.3)
    elif(sys.argv[1] == "--test-images"):
        #Initialize a clear output variable
        __clear_output = False
        #Check if we should clear the output
        if("--clear-output" in sys.argv):
            #Set clear output to true
            __clear_output = True
        if("--verify" in sys.argv):
            #Set verify request to true
            __verify = True
        else:
            #Set the verify request to false
            __verify = False
        #Initialize our trainer
        __trainer = Trainer("datasets/", __categories, clear_output=__clear_output)
        #List all images in the input test directory
        for image in os.listdir(sys.argv[2]):
            image = os.path.join("test", image)
            #Test the given image via prediction
            _g  = __trainer.test_image(image, move_after_testing=False, train_after_testing=False)
            #Print out what we assume the photo is
            print("Object in {} is {}".format(image, _g[1]))
            #Check if we should ask for verification
            if(__verify is True):
                #Ask if the result is correct
                __correct = input("Is that correct? (Y/N) ").lower()
                #Check if the user says Y
                if(__correct == "n"):
                    #Ask what the correct category is
                    __correct_category = input("What is the correct category? ({}) ".format(str(__categories)))
                    #Check if the correct category is part of our categories
                    if(__correct_category not in __categories):
                        print("Category is not in categories list")
                    else:
                        #Train on the correct category
                        __trainer.fit_image_to_saved_model(image, __categories.index(__correct_category))
    elif(sys.argv[1] == "--single-image-train"):
        #Get the image path
        __image_path = sys.argv[2]
        #Get the image category
        __category = sys.argv[1]
        #Check if the category exists
        if(__category not in __categories):
            raise Exception("{} is not a valid category. {} are only acceptable categories".format(__category, __categories))
        __category = __categories.index(__category)
        #Check if we should clear the outputs of the program
        if("--clear-output" in sys.argv):
            #Set clear output to true
            __clear_output = True
        else:
            #Set clear output to false
            __clear_output = False
        #Initialize the trainer
        __trainer = Trainer("datasets/", __categories, clear_output=__clear_output)
        #Train on the given image
        __trainer.fit_image_to_saved_model(__image_path, __category)

    #Check if the user wants to crawl google images
    elif(sys.argv[1] == "--crawl-google"):
        #Get the text being searched
        __query = sys.argv[2]
        #Get the download directory
        __download_dir = sys.argv[3]
        #Initialize the crawler
        __crawler = GoogleImagesCrawler()
        #Crawl and download the search query
        __crawler.crawl_and_download(__query, __download_dir)
    else:
        #Show the help menu
        show_help_menu()
except IndexError:
    show_help_menu()
except KeyboardInterrupt:
    print("Killing...")