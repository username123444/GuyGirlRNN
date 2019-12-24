from trainer import *
import math
__c = ["naked", "clothed"]
__trainer = Trainer("datasets/", __c, clear_output=True)
__trainer.train_image_model(epochs=100, train_times=2, dense_layers=[0], layer_sizes=[64], convolusion_layers=[3], batch_size=Trainer.BATCH_SIZE_DEFAULT, validation_split=0.3)
for image in os.listdir("test"):
    _g  = __trainer.test_image(os.path.join("test", image), move_after_testing=False, train_after_testing=False)
    print("Object in {} is {}".format(image, _g[1]))