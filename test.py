from trainer import *
import math
__c = ["girls", "guys"]
__trainer = Trainer("datasets/", __c, clear_output=False)
#__trainer.train_image_model(epochs=100, dense_layers=[0], layer_sizes=[64], convolusion_layers=[3], batch_size=10, validation_split=0.3)
_g  = __trainer.test_image("testguy.jpg")
print(_g[1])